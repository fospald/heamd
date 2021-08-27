/**

\brief heamd



*/

// http://de.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html


//#define TEST_MERGE_ISSUE		// test openmp issue due to kernel bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65589
//#define TEST_DIST_EVAL		// report distance evaluations for fiber generator 

/*
NOTES:
- Pass array from Pyton to C++: http://www.shocksolution.com/python-basics-tutorials-and-examples/boostpython-numpy-example/
- http://scalability.org/?p=5770
- https://software.intel.com/sites/products/documentation/studio/composer/en-us/2011Update/compiler_c/intref_cls/common/intref_avx2_mm256_i32gather_pd.htm
- http://stackoverflow.com/questions/3596622/negative-nan-is-not-a-nan

TODO:
- use https://bitbucket.org/account/user/VisParGroup/projects/UTANS for CT images
- use Frangi ITK filter (probably itk::Hessian3DToVesselnessMeasureImageFilter) for CT images (as in Herrmann paper: Methods for fibre orientation analysis of X-ray tomography images of steel fibre reinforced concrete (SFRC))
- add and check error tolerances for boundary conditions
- adaptive time stepping and Newton step relaxation on non-convergence 
- try http://numba.pydata.org/
- try https://code.google.com/p/blaze-lib/
- check applyDeltaStaggered
- calcStress const
- use correct mixing rule (s. Milton 9.3)
- swap sigma and epsilon (save vtk, get_field function) in dual scheme (viscosity mode)
- G0OperatorFourierStaggered is 4 times the cost of a FFT
- closestFiber is very expensive
- fast phi method not periodic (bug)
- test different convergence checks (what makes sense for basic and cg schemes? ask Matti!)
- Angular Gaussian in 2D
- make use of tr(epsilon)=0
- clear all fibers action
- general output directory
- different materials
- strain energy check
- write escapecodes only if isatty(FILE)
- correct material constants/equations for 2d
- place_fiber periodic
- implement some checks/results from "David A. Jack and Douglas E. Smith: Elastic Properties of Short-fiber Polymer Composites, Derivation and Demonstration of Analytical Forms for Expectation and Variance from Orientation Tensors", Journal of Composite Materials 2008 42: 277

multigrid improvements:
- solve vector poisson eqation instead of 3 scalar equations
- blocking of smoother 2x2x2 blocks will reduce cache misses
- do not check residuals just run a fixed number of vcycles
- or compute residual within the smoother loop
- combine last smoothing with restriction operation
*/

//#define USE_MANY_FFT

#include <Python.h>
#include <fftw3.h>
#ifdef OPENMP_ENABLED
	#include <omp.h>
#endif
#include <unistd.h>
#include <stdint.h>

#ifdef INTRINSICS_ENABLED
	#include <immintrin.h>
#endif

#ifdef IACA_ENABLED
	#include <iacaMarks.h>
#else
	#define IACA_START
	#define IACA_END
#endif

#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/asio.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional/optional.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/multi_array.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp> 
#include <boost/numeric/conversion/bounds.hpp>

#include <boost/numeric/bindings/traits/ublas_vector2.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#include <boost/numeric/bindings/lapack/gesv.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/lapack/sysv.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/exception/diagnostic_information.hpp>
#include <boost/throw_exception.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/predef/other/endian.h>
#include <boost/math/interpolators/cubic_b_spline.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <boost/program_options.hpp>
#include <boost/limits.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>

#define NPY_NO_DEPRECATED_API 7
//#include <numpy/noprefix.h>
#include <numpy/npy_3kcompat.h>

#define PNG_SKIP_SETJMP_CHECK
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#if BOOST_VERSION >= 106800
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/io/write_view.hpp>
#else
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#endif

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#if BOOST_VERSION >= 106500
#include <boost/python/numpy.hpp>
#endif

#undef ITK_ENABLED

#ifdef ITK_ENABLED
#include <itkImage.h>
#include "itkBinaryThinningImageFilter3D.h"
#endif

#include <csignal>
#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <limits>
#include <list>
#include <algorithm>

namespace ptree = boost::property_tree;
namespace ublas = boost::numeric::ublas;
namespace gil = boost::gil;
namespace acc = boost::accumulators;
namespace pt = boost::posix_time;
namespace po = boost::program_options;
namespace lapack = boost::numeric::bindings::lapack;
namespace py = boost::python;

// http://www.cplusplus.com/forum/unices/36461/
#define _DEFAULT_TEXT	"\033[0m"
#define _BOLD_TEXT	"\033[1m"
#define _RED_TEXT	"\033[0;31m"
#define _GREEN_TEXT	"\033[0;32m"
#define _YELLOW_TEXT	"\033[1;33m"
#define _BLUE_TEXT	"\033[0;34m"
#define _WHITE_TEXT	"\033[0;97m"
#define _INVERSE_COLORS	"\033[7m"
#define _CLEAR_EOL	"\033[2K"

#define DEFAULT_TEXT	TTYOnly(_DEFAULT_TEXT)
#define BOLD_TEXT	TTYOnly(_BOLD_TEXT)
#define RED_TEXT	TTYOnly(_RED_TEXT)
#define GREEN_TEXT	TTYOnly(_GREEN_TEXT)
#define YELLOW_TEXT	TTYOnly(_YELLOW_TEXT)
#define BLUE_TEXT	TTYOnly(_BLUE_TEXT)
#define WHITE_TEXT	TTYOnly(_WHITE_TEXT)
#define INVERSE_COLORS	TTYOnly(_INVERSE_COLORS)
#define CLEAR_EOL	TTYOnly(_CLEAR_EOL)


#define PRECISION Logger::instance().precision()

#ifdef DEBUG
	#define DEBP(x) LOG_COUT << x << std::endl
#else
	#define DEBP(x)
#endif

#define STD_INFINITY(T) std::numeric_limits<T>::infinity()


#ifdef __GNUC__
	#define noinline __attribute__ ((noinline))
#else
	#define noinline
#endif


#if 0
class TTYOnly
{
	const char* _code;
public:
	TTYOnly(const char* code) : _code(code) { }
	friend std::ostream& operator<<(std::ostream& os, const TTYOnly& tto);
};

std::ostream& operator<<(std::ostream& os, const TTYOnly& tto)
{
//	if (isatty(fileno(stdout)) && isatty(fileno(stderr))) os << tto._code;
	return os;
}
#else
	#define TTYOnly(x) (x)
#endif


#define unit_mass	1.6605390666050e-27 // kg
#define unit_length	1e-10 // m
#define unit_time	1e-12 // s
#define unit_T	1.0 // K
#define unit_p	1.0e5 // Pa
#define unit_energy  1.602176634e-19 // J (= 1eV)

#define const_kB 1.380649e-23 // J/K


//! Class for logging
class Logger
{
protected:
	typedef boost::iostreams::tee_device<std::ostream, std::ofstream> Tee;
	typedef boost::iostreams::stream<Tee> TeeStream;

	std::size_t _indent;	// current output indentation
	boost::shared_ptr<Tee> _tee;
	boost::shared_ptr<TeeStream> _tee_stream;
	boost::shared_ptr<std::ofstream> _tee_file;
	boost::shared_ptr<std::ostream> _stream;

	static boost::shared_ptr<Logger> _instance;

	void cleanup()
	{
		_stream.reset();
		_tee_stream.reset();
		_tee.reset();
		_tee_file.reset();
	}

public:
	Logger() : _indent(0)
	{
		_stream.reset(new std::ostream(std::cout.rdbuf()));
	}

	Logger(const std::string& tee_filename) : _indent(0)
	{
		setTeeFilename(tee_filename);
	}

	~Logger()
	{
		cleanup();
	}

	void setTeeFilename(const std::string& tee_filename)
	{
		cleanup();
		_tee_file.reset(new std::ofstream(tee_filename.c_str()));
		_tee.reset(new Tee(std::cout, *_tee_file));
		_tee_stream.reset(new TeeStream(*_tee));
		_stream.reset(new std::ostream(_tee_stream->rdbuf()));
	}

	//! Increase indent
	void incIndent() { _indent++; }

	//! Decrease indent
	void decIndent() { _indent = (size_t) std::max(((int)_indent)-1, 0); }

	//! Write indent to stream
	void indent(std::ostream& stream) const
	{
		std::string space(2*_indent, ' ');
		stream.write(space.c_str(), space.size()*sizeof(char));
	}

	std::streamsize precision() const
	{
		return _stream->precision();
	}

	void flush()
	{
		_stream->flush();
	}

	//! Return standard output stream
	std::ostream& cout()
	{
		std::ostream& stream = *_stream;
		stream << DEFAULT_TEXT;
		//stream << _indent;
		indent(stream);
		return stream;
	}

	//! Return error stream
	std::ostream& cerr() const
	{
		std::ostream& stream = std::cerr;
		// indent(stream);
		stream << RED_TEXT << "ERROR: ";
		return stream;
	}

	//! Return warning stream
	std::ostream& cwarn() const
	{
		std::ostream& stream = std::cerr;
		// indent(stream);
		stream << YELLOW_TEXT << "WARNING: ";
		return stream;
	}

	//! Return static instance
	static Logger& instance()
	{
		if (!_instance) {
			_instance.reset(new Logger());
		}

		return *_instance;
	}
};

// Static logger instance
boost::shared_ptr<Logger> Logger::_instance;

// Shortcut macros for cout and cerr
#define LOG_COUT Logger::instance().cout()
#define LOG_CERR Logger::instance().cerr()
#define LOG_CWARN Logger::instance().cwarn()

// Static exception object
boost::shared_ptr<std::exception> _except;


#ifndef OPENMP_ENABLED
void omp_set_nested(bool n) { }
void omp_set_dynamic(bool d) { }
void omp_set_num_threads(int n) {
	if (n > 1) LOG_CWARN << "OpenMP is disabled, only running with 1 thread!" << std::endl;
}
int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }
int omp_get_max_threads() { return 1; }
#endif


//! Set current exception message and print message to cerr
void set_exception(const std::string& msg)
{
	#pragma omp critical
	{
		// ignore multiple exceptions
		if (!_except) {
			_except.reset(new std::runtime_error(msg));
			LOG_CERR << "Exception set: " << msg << std::endl;
		}
	}
}

//! Print backtrace of current function calls
inline void print_stacktrace(std::ostream& stream)
{
	// print stack trace
	void *trace_elems[32];
	int trace_elem_count = backtrace(trace_elems, sizeof(trace_elems)/sizeof(void*));
	char **symbol_list = backtrace_symbols(trace_elems, trace_elem_count);
	std::size_t funcnamesize = 265;
	char* funcname = (char*) malloc(funcnamesize);

	stream << "Stack trace:" << std::endl;

	// iterate over the returned symbol lines. skip the first, it is the
	// address of this function.
	int c = 0;
	for (int i = trace_elem_count-1; i > 1; i--)
	{
		char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		for (char *p = symbol_list[i]; *p; ++p)
		{
			if (*p == '(')
				begin_name = p;
			else if (*p == '+')
				begin_offset = p;
			else if (*p == ')' && begin_offset) {
				end_offset = p;
				break;
			}
		}

		if (begin_name && begin_offset && end_offset && begin_name < begin_offset)
		{
			*begin_name++ = '\0';
			*begin_offset++ = '\0';
			*end_offset = '\0';

			// mangled name is now in [begin_name, begin_offset) and caller
			// offset in [begin_offset, end_offset). now apply
			// __cxa_demangle():

			int status;
			char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
			if (status == 0) {
				funcname = ret; // use possibly realloc()-ed string
				stream << (boost::format("%02d. %s: " _WHITE_TEXT "%s+%s") % c % symbol_list[i] % funcname % begin_offset).str() << std::endl;
			}
			else {
				// demangling failed. Output function name as a C function with
				// no arguments.
				stream << (boost::format("%02d. %s: %s()+%s") % c % symbol_list[i] % begin_name % begin_offset).str() << std::endl;
			}
		}
		else
		{
			// couldn't parse the line? print the whole line.
			stream << (boost::format("%02d. " _WHITE_TEXT "%s") % c % symbol_list[i]).str() << std::endl;
		}

		c++;
	}

	free(symbol_list);
	free(funcname);
}



// Static empty ptree object
static ptree::ptree empty_ptree;


//! Open file for output, truncate file if already exists
inline void open_file(std::ofstream& fs, const std::string& filename)
{
	fs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

	if (fs.fail()) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("error opening '%s': %s") % filename % strerror(errno)).str()));
	}
}


std::string getFormattedTime(long s)
{
    long sec(s % 60);
    long min((s / 60) % 60);
    long h(s / 3600);
    std::ostringstream oss;
    oss << h << ":" << std::setfill('0') << std::setw(2) << min << ":" << std::setw(2) << sec;
    return oss.str();
}


//! Remove common leading whitespace from (multi-line) string
//! \param s string to dedent
//! \return dedented string
std::string dedent(const std::string& s)
{
	std::vector<std::string> lines;
	std::string indent;
	bool have_indent = false;

	boost::split(lines, s, boost::is_any_of("\n"));

	for (std::size_t i = 0; i < lines.size(); i++)
	{
		boost::algorithm::trim_right_if(lines[i], boost::is_any_of(" \t\v\f\r"));

		if (lines[i].size() == 0) continue;

		if (!have_indent)
		{
			std::string trimmed_line = boost::algorithm::trim_left_copy_if(lines[i], boost::is_any_of(" \t\v\f\r"));
			indent = lines[i].substr(0, lines[i].size() - trimmed_line.size());
			have_indent = true;
			continue;
		}
		
		for (std::size_t j = 0; j < std::max(lines[i].size(), indent.size()); j++) {
			if (lines[i][j] != indent[j]) {
				indent = indent.substr(0, j);
				break;
			}
		}
	}

	for (std::size_t i = 0; i < lines.size(); i++) {
		if (lines[i].size() == 0) continue;
		lines[i] = lines[i].substr(indent.size());
	}

	return boost::join(lines, "\n");
}


//! Class for evaluating python code and managing local variables accross evaluations
class PY
{
protected:
	static boost::shared_ptr<PY> _instance;

	py::object main_module, main_namespace;
	py::dict locals;
	bool enabled;

public:
	PY()
	{
		enabled = false;
	}

#if 0
	~PY()
	{
		LOG_COUT << "~PY" << std::endl;
	}
#endif

	//! Execute python code
	//! \param code the python code
	//! \return result of executing the code
	py::object exec(const std::string& code)
	{
		if (enabled) {
			std::string c = dedent(code);
			return py::exec(c.c_str(), main_namespace, locals);
		}

		return py::object();
	}

	//! Evaluate python expression
	//! \param expr the expression string
	//! \return result object
	py::object eval_obj(const std::string& expr)
	{
		if (enabled) {
			std::string e = expr;
			boost::trim(e);
			return py::eval(e.c_str(), main_namespace, locals);
		}

		return py::object();
	}

	//! Evaluate python expression as type T
	//! \param expr the expression string
	//! \return result converted to type T
	template <class T>
	T eval(const std::string& expr)
	{
		if (enabled) {
			std::string e = expr;
			boost::trim(e);
			py::object result = py::eval(e.c_str(), main_namespace, locals);
			T ret = py::extract<T>(result);
			return ret;
		}

		return boost::lexical_cast<T>(expr);
	}

	//! Get string from ptree and eval as python expression to type T
	//! \param pt the ptree
	//! \param prop the path of the property
	//! \param default_value the default value, if the property does not exists
	//! \return the poperty value
	template <class T>
	T get(const ptree::ptree& pt, const std::string& prop, T default_value)
	{
		boost::optional<std::string> value = pt.get_optional<std::string>(prop);

		if (!value) {
			return default_value;
		}

		return eval<T>(*value);
	}

	//! Get string from ptree and eval as python expression to type T
	//! \param pt the ptree
	//! \param prop the path of the property
	//! \return the poperty value
	//! \exception if the property does not exists
	template <class T>
	T get(const ptree::ptree& pt, const std::string& prop)
	{
		boost::optional<std::string> value = pt.get_optional<std::string>(prop);

		if (!value) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Undefined property: %s!") % prop).str()));
		}

		return eval<T>(*value);
	}

	//! Get string from ptree and eval as python expression
	//! \param pt the ptree
	//! \param prop the path of the property
	//! \return the poperty as object
	//! \exception if the property does not exists
	py::object get_obj(const ptree::ptree& pt, const std::string& prop)
	{
		boost::optional<std::string> value = pt.get_optional<std::string>(prop);

		if (!value) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Undefined property: %s!") % prop).str()));
		}

		return eval_obj(*value);
	}

	//! Enable/Disable python evaluation
	//! If python evaluation is disabled strings are cast to the requested type by a lexical cast
	void set_enabled(bool enabled)
	{
		if (this->enabled == enabled) return;
		this->enabled = enabled;
		
		if (enabled)
		{
			// init main namespace
			main_module = py::import("__main__");
			main_namespace = main_module.attr("__dict__");
			py::exec("from math import *", main_namespace);
		}
	}

	//! Check if python evaluation is enabled
	bool get_enabled()
	{
		return this->enabled;
	}

	//! Clear all local variables
	void clear_locals()
	{
		this->locals = py::dict();
	}

	//! Add a local variable
	//! \param key the variable name
	//! \param value the variable value
	void add_local(const std::string& key, const py::object& value)
	{
		this->locals[key] = value;
	}

	//! Remove a local variable
	//! \param key the variable name
	void remove_local(const std::string& key)
	{
		py::api::delitem(this->locals, key);
	}

	//! Get a local variable
	//! \param key the variable name
	py::object get_local(const std::string& key)
	{
		return this->locals.get(key);
	}

	//! Return static instance of class
	static PY& instance()
	{
		if (!_instance) {
			_instance.reset(new PY());
		}

		return *_instance;
	}

	//! Check if there is an instance
	static bool has_instance()
	{
		return (bool) _instance;
	}

	//! Release static instance of class
	static void release()
	{
		_instance.reset();
	}
};

// Static instance of PY class
boost::shared_ptr<PY> PY::_instance;


//! Shortcut for getting a property from a ptree with python evaluation
//! \param pt the ptree
//! \param prop the property path
//! \return the property
template <class T>
T pt_get(const ptree::ptree& pt, const std::string& prop)
{
	return PY::instance().get<T>(pt, prop);
}

//! Shortcut for getting a property from a ptree with python evaluation
//! \param pt the ptree
//! \param prop the property path
//! \return the property
py::object pt_get_obj(const ptree::ptree& pt, const std::string& prop)
{
	return PY::instance().get_obj(pt, prop);
}

//! Shortcut for getting a property from a ptree with python evaluation and default value
//! \param pt the ptree
//! \param prop the property path
//! \param default_value the default value, if property was not found
//! \return the property
template <class T>
T pt_get(const ptree::ptree& pt, const std::string& prop, T default_value)
{
	return PY::instance().get<T>(pt, prop, default_value);
}

//! Shortcut for getting a property from a ptree as std::string
//! \param pt the ptree
//! \param prop the property path
//! \return the property
template <>
std::string pt_get(const ptree::ptree& pt, const std::string& prop)
{
	boost::optional<std::string> value = pt.get_optional<std::string>(prop);

	if (!value) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Undefined property: %s!") % prop).str()));
	}

	return *value;
}

//! Shortcut for getting a property from a ptree as std::string with default value
//! \param pt the ptree
//! \param prop the property path
//! \param default_value the default value, if property was not found
//! \return the property
template <>
std::string pt_get(const ptree::ptree& pt, const std::string& prop, std::string default_value)
{
	boost::optional<std::string> value = pt.get_optional<std::string>(prop);

	if (!value) {
		return default_value;
	}

	return *value;
}

//! Format a vector expression as string (for console output)
//! \param v the vector expression
//! \param indent enable line indentation using the current logger indent
//! \return the formatted text
template <class VE>
std::string format(const ublas::vector_expression<VE>& v, bool indent = false)
{
	typedef typename VE::value_type T;
	std::size_t N = v().size();
	std::stringstream ss;

	if (indent) {
		Logger::instance().incIndent();
		Logger::instance().indent(ss);
	}

	ss << "( ";
	for (std::size_t i = 0; i < N; i++) {
		if (i > 0) ss << ", ";
		ss << BLUE_TEXT
			<< std::setiosflags(std::ios::scientific | std::ios::showpoint)
			<< std::setprecision(PRECISION) << std::setw(PRECISION+7) << std::right
			<< v()(i) << DEFAULT_TEXT;
	}
	ss << " )";

	if (indent) {
		Logger::instance().decIndent();
	}

	return ss.str();
}


//! Format a matrix expression as string (for console output), lines are indented using the current logger indent
//! \param m the matrix expression
//! \return the formatted text
template <class ME>
std::string format(const ublas::matrix_expression<ME>& m)
{
	typedef typename ME::value_type T;
	std::size_t N = m().size1();
	std::size_t M = m().size2();

	T small = boost::numeric::bounds<T>::smallest();
	T large = boost::numeric::bounds<T>::highest();
	int logmax = std::log10(small), logmin = -logmax;

	for (std::size_t i = 0; i < N; i++) {
		for (std::size_t j = 0; j < M; j++) {
			int lg = std::log10(small + std::min(std::fabs(m()(i, j)), large));
			logmax = std::max(logmax, lg);
			logmin = std::min(logmin, lg);
		}
	}

	std::stringstream ss;

	Logger::instance().incIndent();
	for (std::size_t i = 0; i < N; i++) {
		ss << std::endl;
		Logger::instance().indent(ss);
		for (std::size_t j = 0; j < M; j++) {
			if (j > 0) ss << " ";
			int lg = std::log10(small + std::min(std::fabs(m()(i, j)), large));
			int color = 255 - std::min(logmax - lg, 10)*2;
			ss << "\x1b[38;5;" << color << "m" << TTYOnly(i == j ? _INVERSE_COLORS : "")
				<< std::setiosflags(std::ios::scientific | std::ios::showpoint)
				<< std::setprecision(PRECISION) << std::setw(PRECISION+7) << std::right
				<< m()(i, j) << DEFAULT_TEXT;
		}
	}
	Logger::instance().decIndent();

	//std::cout << "'" << ss.str() << "'" << std::endl;

	return ss.str();
}


//! Format a 3d tensor as string (for console output)
//! \param data pointer to the data, adressed as data[i*ny*nzp + j*nzp + k] for (i,j,k)-th element
//! \param nx number of elements in x (index i)
//! \param ny number of elements in y (index j)
//! \param nz number of elements in z (index k)
//! \param nzp number of elements in z including padding
//! \return the formatted text
template <typename T>
std::string format(const T* data, std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nzp)
{
	ublas::matrix<T> A(ny, nx);
	std::string s;

	for (std::size_t kk = 0; kk < nz; kk++) {
		if (kk > 0) s += "\n";
		for (std::size_t jj = 0; jj < ny; jj++) {
			for (std::size_t ii = 0; ii < nx; ii++) {
				A(jj, ii) = data[ii*ny*nzp + jj*nzp + kk];
			}
		}
		s += format(A) + "\n";
	}

	return s;
}


//! Compute square of 2-norm of vector
//! \param v the vector
//! \return the norm
template <typename T>
inline T norm_2_sqr(const ublas::vector<T>& v)
{
	return ublas::inner_prod(v, v);
}


//! Set components of a vector
//! \param v the vector
//! \param x0 the value for v[0]
//! \param x1 the value for v[1]
//! \param x2 the value for v[2]
template <class V, typename T>
inline void set_vector(V& v, T x0, T x1, T x2)
{
	if (v.size() >= 1) v[0] = x0;
	if (v.size() >= 2) v[1] = x1;
	if (v.size() >= 3) v[2] = x2;
}

//! Set components of a vector
//! \param attr ptree with settings
//! \param v the vector
//! \param name0 settings name for component 0
//! \param name1 settings name for component 1
//! \param name2 settings name for component 2
//! \param def0 default value for component 0
//! \param def1 default value for component 1
//! \param def2 default value for component 2
template <class V, typename T>
inline void read_vector(const ptree::ptree& attr, V& v, const char* name0, const char* name1, const char* name2, T def0, T def1, T def2)
{
	if (v.size() >= 1) v[0] = pt_get<T>(attr, name0, def0);
	if (v.size() >= 2) v[1] = pt_get<T>(attr, name1, def1);
	if (v.size() >= 3) v[2] = pt_get<T>(attr, name2, def2);
}

//! Set components of a matrix
//! \param attr ptree with settings
//! \param m the matrix
//! \param prefix prefix for the component names "prefix%d%d"
//! \param symmetric set to true for symmetric matrix
template <typename T>
inline void read_matrix(const ptree::ptree& attr, ublas::matrix<T>& m, const std::string& prefix, bool symmetric)
{
	for (std::size_t i = 0; i < m.size1(); i++) {
		for (std::size_t j = 0; j < m.size2(); j++) {
			std::string name = (((boost::format("%s%d%d") % prefix) % (i+1)) % (j+1)).str();
			boost::optional< const ptree::ptree& > a = attr.get_child_optional(name);
#if 0
			if (!a && i == j) {
				name = (((boost::format("%s%d") % prefix) % (i+1))).str();
				a = attr.get_child_optional(name);
			}
#endif
			if (a) {
				m(i,j) = pt_get<T>(attr, name, m(i,j));
				if (symmetric) m(j,i) = m(i,j);
			}
		}
	}
}

//! Set components of a Voigt vector
//! \param attr ptree with settings
//! \param v the vector
//! \param prefix prefix for the component names "prefix%d"
template <typename T>
inline void read_voigt_vector(const ptree::ptree& attr, ublas::vector<T>& v, const std::string& prefix)
{
	const std::size_t voigt_indices[9] = {11, 22, 33, 23, 13, 12, 32, 31, 21};

	for (std::size_t i = 0; i < 3; i++) {
		v(i) = pt_get<T>(attr, ((boost::format("%s%d") % prefix) % (i+1)).str(), v(i));
	}

	for (std::size_t i = 0; i < v.size(); i++) {
		v(i) = pt_get<T>(attr, ((boost::format("%s%d") % prefix) % voigt_indices[i]).str(), v(i));
	}
}


//! Matrix inversion routine. Uses gesv in uBLAS to invert a matrix.
//! \param input input matrix
//! \param inverse inverse ouput matrix
template<typename T, int DIM>
void InvertMatrix(const ublas::c_matrix<T,DIM,DIM>& input, ublas::c_matrix<T,DIM,DIM>& inverse)
{
#if 1
	ublas::c_matrix<T,DIM,DIM> icopy(input);
	inverse.assign(ublas::identity_matrix<T>(input.size1()));
	int res = lapack::gesv(icopy, inverse);
	if (res != 0) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Matrix inversion failed for matrix:\n%s!") % format(input)).str()));
	}
#else
	typedef ublas::permutation_matrix<std::size_t> pmatrix;

	// create a working copy of the input
	ublas::c_matrix<T,DIM,DIM> A(input);

	// create a permutation matrix for the LU-factorization
	pmatrix pm(A.size1());

	// perform LU-factorization
	int res = ublas::lu_factorize(A, pm);
	if (res != 0) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("LU factorization failed for matrix:\n%s!") % format(input)).str()));
	}

	// create identity matrix of "inverse"
	inverse.assign(ublas::identity_matrix<T>(A.size1()));

	// backsubstitute to get the inverse
	ublas::lu_substitute(A, pm, inverse);
#endif
}


//! A progress bar for console output
template <typename T>
class ProgressBar
{
public:

	//! Create progress bar with maximum value and a number of update steps
	//! \param max the maximum value for the progress parameter
	//! \param steps number of update steps (the number of times the progress text is updated)
	ProgressBar(T max = 100, T steps = 100)
		: _max(max), _dp(max/steps), _p(0), _p_old(-max)
	{
	}

	//! Increment the progress by one
	//! \return true if you should print the progress message
	bool update()
	{
		return update(_p + 1);
	}

	//! Update the progress to value p
	//! \return true if you should print the progress message
	bool update(T p)
	{
		_p = std::min(std::max(p, (T)0), _max);
		return (std::abs(_p - _p_old) > _dp) || complete();
	}

	//! Returns true if the progress is complete
	bool complete() const
	{
		return (_p >= _max);
	}

	//! Returns text for the end of the progress message, i.e.
	//! cout << pb.message() << "saving..." << pb.end();
	const char* end()
	{
		Logger::instance().flush();

		if (complete()) {
			return _DEFAULT_TEXT _CLEAR_EOL "\n";
		}
		return _DEFAULT_TEXT _CLEAR_EOL "\r";
	}

	//! Returns the progress in percent
	T progress() const
	{
		return _p/_max*100;
	}

	//! Returns the current progress message as stream to cout
	std::ostream& message()
	{
		T percent = progress();
		_p_old = _p;
		std::ostream& stream = LOG_COUT;
		stream << (complete() ? GREEN_TEXT : YELLOW_TEXT) << (boost::format("%.2f%% complete: ") % percent);
		return stream;
	}

protected:
	T _max, _dp, _p, _p_old;
};



//! Class for measuring elasped time between construction and destruction and console output of timings
class Timer
{
protected:
	class Info {
	public:
		Info() : calls(0), total_duration() { }
		std::size_t calls;
		pt::time_duration total_duration;
	};

	typedef std::map<std::string, boost::shared_ptr<Info> > InfoMap;
	static InfoMap info_map;
	
	std::string _text;
	bool _print, _log;
	pt::ptime _t0;
	boost::shared_ptr<Info> _info;

public:
	Timer() : _print(false)
	{
		start();
	}

	//! Constructor. The timer is started automatically.
	//! \param text text for display in the console, when timer starts/finishes and the statistics
	//! \param print enable console output 
	//! \param log enable logging for statistics of function calls etc. 
	Timer(const std::string& text, bool print = true, bool log = true) : _text(text), _print(print), _log(log)
	{
#ifdef DEBUG
		_print = true;
#endif

		if (_print) {
			LOG_COUT << BOLD_TEXT << "Begin " << _text << std::endl;
			Logger::instance().incIndent();
		}
		start();

		if (_log) {
			InfoMap::iterator item = Timer::info_map.find(text);
			if (item != Timer::info_map.end()) {
				_info = item->second;
			}
			else {
				_info.reset(new Info());
				Timer::info_map.insert(InfoMap::value_type(text, _info));
			}
		}
	}

	//! Start the timer
	void start()
	{
		_t0 = pt::microsec_clock::universal_time();
	}
	
	//! Return current elaspsed time
	pt::time_duration duration()
	{
		pt::ptime t = pt::microsec_clock::universal_time();
		return (t - _t0);
	}

	//! Return current elasped time in seconds
	double seconds()
	{
		return duration().total_milliseconds()*1e-3;
	}

	//! Return current elasped time in seconds
	operator double() { return seconds(); }

	~Timer()
	{
		pt::time_duration dur = duration();

		if (_print) {
			Logger::instance().decIndent();
			LOG_COUT << BOLD_TEXT << "Finished " << _text << " in " << (dur.total_milliseconds()*1e-3) << "s" << std::endl;
		}

		if (_info) {
			_info->calls ++;
			_info->total_duration += dur;
		}
	}

	//! Print statistics
	static void print_stats();

	//! Clear statistics
	static void reset_stats();
};


Timer::InfoMap Timer::info_map;


void Timer::print_stats()
{
	std::size_t title_width = 8;
	std::size_t calls_width = 5;
	std::size_t time_width = 9;
	std::size_t time_per_call_width = 9;
	std::size_t relative_width = 9;
	pt::time_duration total;

	for (Timer::InfoMap::iterator i = Timer::info_map.begin(); i != Timer::info_map.end(); ++i) {
		Timer::Info& info = *(i->second);
		if (info.calls == 0) continue;
		title_width = std::max(title_width, i->first.length());
		total += info.total_duration;
		calls_width = std::max(calls_width, 1+(std::size_t)std::log10((double)info.calls));
	}

	double sec_total = total.total_milliseconds()*1e-3;

	LOG_COUT << (boost::format("%s|%s|%s|%s|%s\n")
		% boost::io::group(std::setw(title_width), "Function")
		% boost::io::group(std::setw(calls_width), "Calls")
		% boost::io::group(std::setw(time_width), "Time")
		% boost::io::group(std::setw(time_per_call_width), "Time/Call")
		% boost::io::group(std::setw(relative_width), "Relative")
	).str();

	LOG_COUT << std::string(title_width, '=')
		+ "+" + std::string(calls_width, '=')
		+ "+" + std::string(time_width, '=')
		+ "+" + std::string(time_per_call_width, '=')
		+ "+" + std::string(relative_width, '=')
		+ "\n";

	for (Timer::InfoMap::iterator i = Timer::info_map.begin(); i != Timer::info_map.end(); ++i) {
		Timer::Info& info = *(i->second);
		if (info.calls == 0) continue;
		double sec = info.total_duration.total_milliseconds()*1e-3;
		LOG_COUT << (boost::format("%s|%d|%g|%g|%g\n")
			% boost::io::group(std::setw(title_width), i->first)
			% boost::io::group(std::setw(calls_width), info.calls)
			% boost::io::group(std::setw(time_width), std::setprecision(4), sec)
			% boost::io::group(std::setw(time_per_call_width), std::setprecision(4), sec/info.calls)
			% boost::io::group(std::setw(relative_width), std::setprecision(4), sec/sec_total*100.0)
		).str();
	}

	LOG_COUT << std::string(title_width, '=')
		+ "+" + std::string(calls_width, '=')
		+ "+" + std::string(time_width, '=')
		+ "+" + std::string(time_per_call_width, '=')
		+ "+" + std::string(relative_width, '=')
		+ "\n";
	
	LOG_COUT << (boost::format("%s|%s|%4g|%s|%4g\n")
		% boost::io::group(std::setw(title_width), "total")
		% boost::io::group(std::setw(calls_width), "-")
		% boost::io::group(std::setw(time_width), std::setprecision(4), sec_total)
		% boost::io::group(std::setw(time_per_call_width), "-")
		% boost::io::group(std::setw(relative_width), std::setprecision(4), 100.0)
	).str();
}


void Timer::reset_stats()
{
	info_map.clear();
}



//! Standard normal (mu=0, sigma=1) distributed number generator
template<typename T>
class RandomNormal01
{
	// random number generator stuff
	typedef boost::normal_distribution<T> NumberDistribution; 
	typedef boost::mt19937 RandomNumberGenerator; 
	NumberDistribution _distribution; 
	RandomNumberGenerator _generator; 
	boost::variate_generator<RandomNumberGenerator&, NumberDistribution> _rnd; 
	static RandomNormal01<T> _instance;

public:
	
	// constructor
	RandomNormal01() :
		_distribution(0, 1), _generator(), _rnd(_generator, _distribution)
	{
	}
	
	//! change random seed
	void seed(int s) {
		// http://stackoverflow.com/questions/4778797/setting-seed-boostrandom
		_rnd.engine().seed(s);
		_rnd.distribution().reset();
	}
	
	//! return random number
	T rnd() { return _rnd(); }
	
	//! return static instance
	static RandomNormal01& instance() { return _instance; }	
};

template<typename T>
RandomNormal01<T> RandomNormal01<T>::_instance;


//! Uniform number generator in [0,1]
template<typename T>
class RandomUniform01
{
	// random number generator stuff
	typedef boost::uniform_real<T> NumberDistribution; 
	typedef boost::mt19937 RandomNumberGenerator; 
	NumberDistribution _distribution; 
	RandomNumberGenerator _generator; 
	boost::variate_generator<RandomNumberGenerator&, NumberDistribution> _rnd; 
	static RandomUniform01<T> _instance;

public:
	
	// constructor
	RandomUniform01() :
		_distribution(0, 1), _generator(), _rnd(_generator, _distribution)
	{
	}
	
	//! change random seed
	void seed(int s) {
		// http://stackoverflow.com/questions/4778797/setting-seed-boostrandom
		_rnd.engine().seed(s);
		_rnd.distribution().reset();
	}
	
	//! return random number
	T rnd() { return _rnd(); }
	
	//! return static instance
	static RandomUniform01& instance() { return _instance; }	
};

template<typename T>
RandomUniform01<T> RandomUniform01<T>::_instance;









template <typename T, int DIM>
class Element
{
public:
	T r;	// radius
	T m;	// mass
	std::size_t 	Z;	// atomic number
	std::string 	color;
	std::string 	id;

	Element()
	{
		r = 1.0;
		m = 1.0;
		Z = 1.0;
		color = "#ffffff";
		id = "H";
	}
};


template <typename T, int DIM>
class ElementDatabase
{
protected:
	typedef std::map<std::string, boost::shared_ptr<Element<T,DIM> > > ElementMap;

	ElementMap _map;

public:
	// reads element objects form xml file
	void load(const std::string& filename)
	{
		LOG_COUT << "load element file  " << filename << std::endl;
		ptree::ptree xml_root;
		read_xml(filename, xml_root, 0*ptree::xml_parser::trim_whitespace);
		const ptree::ptree& elements = xml_root.get_child("elements", empty_ptree);
		this->load_xml(elements);
	}

	void load_xml(const ptree::ptree& elements)
	{
		BOOST_FOREACH(const ptree::ptree::value_type &v, elements)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "element")
			{
				std::string id = pt_get<std::string>(attr, "id", "");
				boost::shared_ptr<Element<T,DIM> > element;

				if (_map.count(id) == 0) {
					//LOG_COUT << "new element " << id << std::endl;
					element.reset(new Element<T,DIM>());
					element->id = id;
					_map[id] = element;
				}
				else {
					//LOG_COUT << "update element " << id << std::endl;
					element = _map[id];
				}

				element->Z = pt_get<std::size_t>(attr, "Z", element->Z);
				element->m = pt_get<T>(attr, "m", element->m/unit_mass) * unit_mass;
				element->r = pt_get<T>(attr, "r", element->r/unit_length) * unit_length;
				element->color = pt_get<std::string>(attr, "color", element->color);
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown element: '%s'") % v.first).str()));
			}
		}
	}

	boost::shared_ptr<Element<T,DIM> > get(const std::string& key)
	{
		return _map[key];
	}
};





template <typename T, int DIM>
class Molecule
{
public:
	ublas::c_vector<T, DIM> x;	// position
	ublas::c_vector<T, DIM> Fn;	// force by normal molecules
	ublas::c_vector<T, DIM> Fg;	// force by ghost molecules
	ublas::c_vector<T, DIM> F;	// force (Fn + Fg)
	ublas::c_vector<T, DIM> F0;	// previous force
	ublas::c_vector<T, DIM> v;	// velocity
	T U;				// potential energy
	boost::shared_ptr<Element<T, DIM> > element;	// base element
	std::vector<int> nn_indices;
};


template <typename T, int DIM>
struct GhostMolecule
{
	ublas::c_vector<T, DIM> t;	// translation
	std::size_t molecule_index;    // base molecule 
};


template <typename T, int DIM>
class UnitCell
{
	// init dimensions from settings
	
	// methods
	// scale by factor
	// get dimensions
	// generate random point uniformly distributed
	// generate ghost points for a point
	// wrap point (to stay within the cell periodically)
	// get list of translations to neighbour cells
	// get bounding box

public:
	virtual void readSettings(const ptree::ptree& pt) = 0;

	virtual void wrap_vector(ublas::c_vector<T, DIM>& v) const = 0;
	virtual const ublas::c_vector<T, DIM>& bb_origin() const = 0;
	virtual const ublas::c_vector<T, DIM>& bb_size() const = 0;
	virtual T volume() const = 0;
	virtual void scale(T s) = 0;

	virtual const std::vector< ublas::c_vector<T, DIM> >& neighbour_translations() const = 0;
};


template <typename T, int DIM>
class CubicUnitCell : public UnitCell<T, DIM>
{
protected:
	// cell dimensions
	ublas::c_vector<T, DIM> _p;
	ublas::c_vector<T, DIM> _a;
	std::vector< ublas::c_vector<T, DIM> > _neighbour_translations;

public:
	CubicUnitCell()
	{
		for (int d = 0; d < DIM; d++) {
			_p[d] = 0.0 * unit_length;
			_a[d] = 1.0 * unit_length;
		}
	}

	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		for (int d = 0; d < DIM; d++) {
			_p[d] = pt_get(pt, (boost::format("p%s") % d).str(), _p[d]/unit_length) * unit_length;
			_a[d] = pt_get(pt, (boost::format("a%s") % d).str(), _a[d]/unit_length) * unit_length;
		}

		// compute neighbour translations
		std::size_t m = std::pow(3, DIM);
		for (std::size_t j = 0; j < m; j++)
		{
			ublas::c_vector<T, DIM> t;
			std::size_t k = j;
			for (std::size_t d = 0; d < DIM; d++) {
				std::size_t index = k % 3;
				k -= index;
				k /= 3;
				t[d] = _a[d]*(T)((int)index - 1);
			}

			if (ublas::inner_prod(t, t) == (T)0) continue;
			
			_neighbour_translations.push_back(t);

			//LOG_COUT << "neighbour translation " << t[0] << " " << t[1] << " " << t[2] << std::endl;
		}
	}

	void wrap_vector(ublas::c_vector<T, DIM>& v) const
	{
		ublas::c_vector<T, DIM> n;

		for (int d = 0; d < DIM; d++) {
			n[d] = (v[d] - _p[d])/_a[d];
			v[d] -= std::floor(n[d])*_a[d];
		}
	}

	const ublas::c_vector<T, DIM>& bb_origin() const { return _p; }
	const ublas::c_vector<T, DIM>& bb_size() const { return _a; }

	T volume() const
	{
		T V = _a[0];
		for (int d = 1; d < DIM; d++) {
			V *= _a[d];
		}
		return V;
	}

	void scale(T s)
	{
		if (s == 1.0) return;

		for (int d = 0; d < DIM; d++) {
			_a[d] *= s;
		}
		
		for (std::size_t i = 0; i < _neighbour_translations.size(); i++) {
			_neighbour_translations[i] *= s;
		}
	}

	const std::vector< ublas::c_vector<T, DIM> >& neighbour_translations() const { return _neighbour_translations; }
};


template <typename T, int DIM>
class VerletCell
{
protected:
	std::vector<int> _indices;	// list with molecule indices
	std::vector<int> _nindices;	// list with neighbour molecule indices

public:
	VerletCell()
	{
	}

	const std::vector<int>& indices() const {
		return _indices;
	}

	std::vector<int>& nindices() {
		return _nindices;
	}

	void clear()
	{
		_indices.clear();
		_nindices.clear();
	}

	void add(int index)
	{
		_indices.push_back(index);
	}
};


template <typename T, int DIM>
class VerletMap
{
protected:
	typedef boost::multi_array< boost::shared_ptr<VerletCell<T, DIM> >, DIM> array_type;

	boost::shared_ptr< array_type > _cells;
	std::array<std::size_t, DIM> _dims;

public:
	VerletMap(const std::array<std::size_t, DIM> & dims)
	{
		_dims = dims;

		// init array
		_cells.reset(new array_type(dims));

		// create verlet cells
		auto elements = boost::make_iterator_range(_cells->data(), _cells->data() + _cells->num_elements());
		for (auto& element : elements) {
			element.reset(new VerletCell<T, DIM>());
		}
	}

	void clear()
	{
		auto elements = boost::make_iterator_range(_cells->data(), _cells->data() + _cells->num_elements());
		for (auto& element : elements) {
			element->clear();
		}
	}

	void print()
	{
		int k = 0;
		auto elements = boost::make_iterator_range(_cells->data(), _cells->data() + _cells->num_elements());
		for (auto& element : elements) {
			const std::vector<std::size_t>& indices = element->indices();
			LOG_COUT << "verlet cell " << k << " ";
			for (std::size_t i = 0; i < indices.size(); i++) {
				LOG_COUT << " " << indices[i];
			}
			LOG_COUT << std::endl;
			k++;
		}
	}

	boost::shared_ptr<VerletCell<T, DIM> > get_cell(const std::array<std::size_t, DIM>& index)
	{
		return (*_cells)(index);
	}

/*
	{
  A(idx) = 3.14;
  assert(A(idx) == 3.14);

	}
*/
};

template <typename D, int DIM>
class SimulationInterval
{
public:
	D t, T, p, V, dt;
	bool enforce_T, enforce_mean_T;
};


template <typename T, int DIM>
class ElementFraction
{
public:
	boost::shared_ptr<Element<T,DIM> > element;
	T fraction;
	std::size_t n;
};


template <typename T, int DIM>
class PairPotential
{
public:
	virtual void readSettings(const ptree::ptree& pt) = 0;

	virtual void compute(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params) = 0;

	virtual std::size_t num_interpolation_params() const = 0;
	virtual void U_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params) const = 0;
	virtual T dU_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params, const std::vector<T>& dparams) const = 0;
};


template <typename T, int DIM>
class InterpolatedPotential : public PairPotential<T, DIM>
{
protected:
	typedef struct
	{
		std::vector< boost::shared_ptr< boost::math::cubic_b_spline<T> > > ip;
		std::vector< std::vector<T> > f;
	} InterpolationParams;

	typedef std::map<std::string, boost::shared_ptr<InterpolationParams> > ParamMap1;
	typedef std::map<std::string, ParamMap1> ParamMap2;
	
	std::list< boost::shared_ptr<InterpolationParams> > _param_list;
	ParamMap2 _param_map;
	boost::shared_ptr< PairPotential<T, DIM> > _potential;
	T _r_min;
	T _r_max;
	std::size_t _n;

	void initInterpolators(std::list< boost::shared_ptr<Element<T,DIM> > >& elements)
	{
		auto it1 = elements.begin();
		while (it1 != elements.end())
		{
			auto it2 = elements.begin();
			while (it2 != elements.end())
			{
				addInterpolator(*it1, *it2);
				++it2;
			}

			++it1;
		}
	}

	void addInterpolator(boost::shared_ptr<Element<T,DIM> > el1, boost::shared_ptr<Element<T,DIM> > el2)
	{
		boost::shared_ptr<InterpolationParams> params;

		typename ParamMap2::iterator parami1 = _param_map.find(el1->id);
		if (parami1 == _param_map.end()) {
			_param_map.insert(typename ParamMap2::value_type(el1->id, ParamMap1()));
			parami1 = _param_map.find(el1->id);
		}
		typename ParamMap1::iterator parami2 = parami1->second.find(el2->id);
		if (parami2 != parami1->second.end()) {
			return;
		}
		
		params.reset(new InterpolationParams());
		parami1->second.insert(typename ParamMap1::value_type(el2->id, params));
		parami2 = parami1->second.find(el2->id);

		std::vector<std::size_t> nn_indices(1);
		std::vector<int> nmolecule_indices(1);
		std::vector<T> dist2(1);
		std::vector< ublas::c_vector<T, DIM> > dir(1);
		std::vector< boost::shared_ptr< Molecule<T, DIM> > > molecules;
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > > gmolecules;

		boost::shared_ptr< Molecule<T, DIM> > m1, m2;

		m1.reset(new Molecule<T, DIM>());
		m1->element = el2;
		m2.reset(new Molecule<T, DIM>());
		m2->element = el1;
		molecules.push_back(m1);
		molecules.push_back(m2);
		
		nn_indices[0] = 0;
		nmolecule_indices[0] = 0;
		std::fill(dist2.begin(), dist2.end(), (T)0);
		std::fill(m1->x.begin(), m1->x.end(), (T)0);
		std::fill(m2->x.begin(), m2->x.end(), (T)0);
		std::fill(dir[0].begin(), dir[0].end(), (T)0);

		std::size_t np = _potential->num_interpolation_params();
		std::vector< std::vector<T> > f(np);
		std::vector<T> ip_params(np);
		std::vector<T> U;
		T dr = (_r_max - _r_min) / (_n - 1);

		for (std::size_t i = 0; i < _n; i++)
		{
			T r = _r_min + dr*i;
			m2->x[0] = r;
			dir[0][0] = -r;
			dist2[0] = r*r;
			_potential->compute(1, nn_indices, nmolecule_indices, dist2, dir, molecules, gmolecules, ip_params);

			for (std::size_t k = 0; k < np; k++) {
				f[k].push_back(ip_params[k]);
			}

			U.push_back(m2->U);

			//LOG_COUT << "i=" << i << " U=" << (boost::format("%g") % U[i]).str() << std::endl;
		}

		params->f = f;
		params->ip.resize(np);

		// check if we have the same interpolation function
		// if so then use it to save cache memory
		auto it = _param_list.begin();
		while (it != _param_list.end())
		{
			for (std::size_t k = 0; k < np; k++) {
				if (params->ip[k]) continue;
				if ((*it)->f[k] == f[k]) {
					params->ip[k] = (*it)->ip[k];
				}
			}
			++it;
		}

		for (std::size_t k = 0; k < np; k++) {
			//LOG_COUT << "id=" << el1->id << " rmin=" << _r_min << " dr=" << dr << " " << f[k][0] << " " << f[k][_n-1] << std::endl;
			if (params->ip[k]) continue;
			params->ip[k].reset(new boost::math::cubic_b_spline<T>(
				f[k].begin(), f[k].end(), _r_min, dr,
				std::numeric_limits<T>::quiet_NaN(),
				std::numeric_limits<T>::quiet_NaN())
			);
		}

		if (1) {
			std::string filename = "potential_" + el1->id + "_" + el2->id + ".csv";
			if (!boost::filesystem::exists(filename)) {
				std::ofstream s;
				s.open(filename);
				for (std::size_t i = 0; i < _n; i++) {
					T r = _r_min + dr*i;
					s << (boost::format("%d\t%g\t%g") % i % (r/unit_length) % (U[i]/unit_energy)).str();
					
					for (std::size_t k = 0; k < np; k++) {
						ip_params[k] = ((*(params->ip[k]))(r));
					}
					
					U_from_interpolation_params(*m2, ip_params);
					s << (boost::format("\t%g") % (m2->U/unit_energy)).str();

					for (std::size_t k = 0; k < np; k++) {
						s << (boost::format("\t%g\t%g\t%g") % (f[k][i]) % ip_params[k] % ((*(params->ip[k])).prime(r))).str();
					}
					s << "\n";
				}
				s.close();
			}
		}


		_param_list.push_back(params);
	}

	inline boost::shared_ptr<InterpolationParams> getInterpolator(boost::shared_ptr<Element<T,DIM> > el1, boost::shared_ptr<Element<T,DIM> > el2)
	{
		return _param_map[el1->id][el2->id];
	}

public:
	InterpolatedPotential(boost::shared_ptr< PairPotential<T, DIM> > potential, T r_min, T r_max, std::size_t n, std::list< boost::shared_ptr<Element<T,DIM> > >& elements)
	{
		_potential = potential;
		_r_min = r_min;
		_r_max = r_max;
		_n = n;
		initInterpolators(elements);
	}

	void readSettings(const ptree::ptree& pt)
	{
	}

	void compute(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params)
	{
		ublas::c_vector<T, DIM> F;
		ublas::c_vector<T, DIM> Fn_sum, Fg_sum;

		std::size_t np = num_interpolation_params();

		std::fill(Fn_sum.begin(), Fn_sum.end(), (T)0);
		std::fill(Fg_sum.begin(), Fg_sum.end(), (T)0);
		std::fill(ip_params.begin(), ip_params.end(), (T)0);
		std::vector<T> dip_params(np);
		
		auto nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			int mi0 = nmolecule_indices[*nn];

			if (mi0 < 0) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			boost::shared_ptr<InterpolationParams> params = getInterpolator(molecules[mi]->element, molecules[mi0]->element);

			T r = std::sqrt(dist2[*nn]);

			if (r < _r_min || r > _r_max) {
				LOG_COUT << "interpolation outside r=" << r << " rmin=" << _r_min << " rmax=" << _r_max << std::endl;
			}

			for (std::size_t k = 0; k < np; k++) {
				ip_params[k] += (*(params->ip[k]))(r);
			}

			++nn;
		}

		_potential->U_from_interpolation_params(*molecules[mi], ip_params);
		
		nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			int mi0 = nmolecule_indices[*nn];
			bool is_ghost = mi0 < 0;

			if (is_ghost) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			boost::shared_ptr<InterpolationParams> params = getInterpolator(molecules[mi]->element, molecules[mi0]->element);

			T r = std::sqrt(dist2[*nn]);

			for (std::size_t k = 0; k < np; k++) {
				dip_params[k] = (*(params->ip[k])).prime(r);
			}

			T dU = _potential->dU_from_interpolation_params(*molecules[mi], ip_params, dip_params);

			F = (-dU/r)*dir[*nn];

			if (is_ghost)
			{
				#pragma omp critical
				molecules[mi0]->Fg += F;

				Fg_sum += F;
			}
			else
			{
				#pragma omp critical
				molecules[mi0]->Fn += F;

				Fn_sum += F;
			}

			++nn;
		}

		#pragma omp critical
		{
			molecules[mi]->Fn -= Fn_sum;
			molecules[mi]->Fg -= Fg_sum;
		}
	}

	std::size_t num_interpolation_params() const
	{
		return _potential->num_interpolation_params();
	}

	void U_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params) const
	{
		_potential->U_from_interpolation_params(m, params);
	}

	T dU_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params, const std::vector<T>& dparams) const
	{
		return _potential->dU_from_interpolation_params(m, params, dparams);
	}

};




template <typename T, int DIM>
class EmbeddedAtomPotential : public PairPotential<T, DIM>
{
protected:
	typedef struct
	{
		T re, fe, rhoe, rhos, alpha, beta, A, B, cai, ramda, Fi0, Fi1, Fi2, Fi3, Fm0, Fm1, Fm2, Fm3, fnn, Fn;
		T ielement, amass, Fm4, beta1, ramda1, rhol, rhoh;
		T blat, rhoin, rhoout;
	} ElementParams;

	typedef std::map<std::string, boost::shared_ptr<ElementParams> > ParamMap;
	
	ParamMap _param_map;

	inline T densFunc(T r, T b, T c)
	{
		return std::exp(-b*(r - 1.0)) / (1.0 + std::pow(std::max((T)0, r - c), 20.0));
	}

	inline T ddensFunc(T r, T b, T c)
	{
		return -densFunc(r, b, c) * (b + (r > c ? (20*std::pow(r - c, 19.0) / (1.0 + std::pow(std::max((T)0, r - c), 20.0))) : 0));
	}

	void addElement(std::string element_name, boost::shared_ptr<ElementParams> params)
	{
		// the units in the file are Angstrom and eV and need to be converted to SI units
		params->re *= unit_length;
		params->A *= unit_energy;
		params->B *= unit_energy;
		params->Fi0 *= unit_energy;
		params->Fi1 *= unit_energy;
		params->Fi2 *= unit_energy;
		params->Fi3 *= unit_energy;
		params->Fm0 *= unit_energy;
		params->Fm1 *= unit_energy;
		params->Fm2 *= unit_energy;
		params->Fm3 *= unit_energy;
		params->Fm4 *= unit_energy;
		params->Fn *= unit_energy;

		params->blat = std::sqrt(2.0)*params->re;
		params->rhoin = params->rhol*params->rhoe;
		params->rhoout = params->rhoh*params->rhoe;

		_param_map.insert(typename ParamMap::value_type(element_name, params));
		//LOG_COUT << "eam " << element_name << std::endl;

		if (params->ramda != params->ramda1) {
			LOG_COUT << "ramda " << element_name << std::endl;
		}
	}

	inline const ElementParams& getParams(std::string& id) const
	{
		typename ParamMap::const_iterator parami = _param_map.find(id);
		if (parami == _param_map.end()) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("No EAM parameters for element id '%s'") % id).str()));
		}
		return *(parami->second);
	}


public:
	void readSettings(const ptree::ptree& pt)
	{
		std::string filename = "../../data/eam_database/parameters";
		boost::filesystem::ifstream fh(filename);
		std::string line;
		std::string element_name;
		std::size_t iparam = 0;
		std::size_t num_params = 27;
		boost::shared_ptr<ElementParams> params;

		while (std::getline(fh, line))
		{
			if (line.length() == 0)
				continue;

			if (std::isalpha(line[0])) {
				if (iparam == num_params) {
					addElement(element_name, params);
				}
				params.reset(new ElementParams());
				element_name = line;
				iparam = 0;
				continue;
			}
			
			T* params_ptr = (T*) params.get();
			params_ptr[iparam] = boost::lexical_cast<T>(line);

			iparam++;
		}

		if (iparam == num_params) {
			addElement(element_name, params);
		}
	}

	void compute_(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params)
	{
		//LOG_COUT << "compute" << std::endl;
		//LOG_COUT << "nn " << (*nn) << std::endl;

		// potential energy

		boost::shared_ptr<ElementParams> p = _param_map[molecules[mi]->element->id];

		//LOG_COUT << "mi: " << mi << std::endl;
		
		T sum_phi = 0, rho = 0;
		auto nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			int mi0 = nmolecule_indices[*nn];

			if (mi0 < 0) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			boost::shared_ptr<ElementParams> p0 = _param_map[molecules[mi0]->element->id];

			T r = std::sqrt(dist2[*nn]);

			//LOG_COUT << "nn " << (*nn) << " r: " << r << std::endl;

			T r_by_re = r/p->re;
			T r_by_re0 = r/p0->re;

			// electron density
			rho += p0->fe*this->densFunc(r_by_re0, p0->beta, p0->ramda1);

			// pair potential sum

			T da = this->densFunc(r_by_re0, p0->beta, p0->ramda);
			T fa = p0->fe*da;
			T phi_aa = p0->A*this->densFunc(r_by_re0, p0->alpha, p0->cai) - p0->B*da;

			T db = this->densFunc(r_by_re, p->beta, p->ramda);
			T fb = p->fe*db;
			T phi_bb = p->A*this->densFunc(r_by_re, p->alpha, p->cai) - p->B*db;

			sum_phi += 0.5*(fb/fa*phi_aa + fa/fb*phi_bb);

			
			++nn;
		}

		//LOG_COUT << "rho: " << rho << " rhoe: " << p->rhoe << std::endl;

		// embedding energy
		T FE;
		if (rho < p->rhoin) {
			T c = rho/p->rhoin - 1.0;
			FE = p->Fi0 + c*(p->Fi1 + c*(p->Fi2 + c*p->Fi3));
		}
		else if (rho < p->rhoout)
		{
			T Fm33 = (rho < p->rhoe) ? p->Fm3 : p->Fm4;
			T c = rho/p->rhoe - 1.0;
			FE = p->Fm0 + c*(p->Fm1 + c*(p->Fm2 + c*Fm33));
		}
		else {
			T c = rho/p->rhos;
			FE = p->Fn*(1.0 - p->fnn*std::log(c))*std::pow(c, p->fnn);
		}

		//LOG_COUT << "U: " << sum_phi  << " " << FE << std::endl;

		// EAM potential
		molecules[mi]->U = 0.5*sum_phi + FE;

	
		// accelerations

		ublas::c_vector<T, DIM> F;
		ublas::c_vector<T, DIM> Fn_sum, Fg_sum;

		std::fill(Fn_sum.begin(), Fn_sum.end(), (T)0);
		std::fill(Fg_sum.begin(), Fg_sum.end(), (T)0);

		nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			// compute potential gradient w.r.t. molecules[nn]->x

			int mi0 = nmolecule_indices[*nn];
			bool is_ghost = mi0 < 0;

			if (is_ghost) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			// get parameters for element
			typename ParamMap::iterator parami = _param_map.find(molecules[mi0]->element->id);
			if (parami == _param_map.end()) {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("No EAM parameters for element id '%s'") % molecules[mi0]->element->id).str()));
			}
			boost::shared_ptr<ElementParams> p0 = parami->second;

			T r = std::sqrt(dist2[*nn]);
			T r_by_re = r/p->re;
			T dr_by_re = 1.0/p->re;
			T r_by_re0 = r/p0->re;
			T dr_by_re0 = 1.0/p0->re;

			// electron density
			T drho = p0->fe*this->ddensFunc(r_by_re0, p0->beta, p0->ramda1)*dr_by_re0;

			// pair potential sum

			T da = this->densFunc(r_by_re0, p0->beta, p0->ramda);
			T dda = this->ddensFunc(r_by_re0, p0->beta, p0->ramda)*dr_by_re0;
			T fa = p0->fe*da;
			T dfa = p0->fe*dda;
			T phi_aa = p0->A*this->densFunc(r_by_re0, p0->alpha, p0->cai) - p0->B*da;
			T dphi_aa = p0->A*this->ddensFunc(r_by_re0, p0->alpha, p0->cai)*dr_by_re0 - p0->B*dda;

			T db = this->densFunc(r_by_re, p->beta, p->ramda);
			T ddb = this->ddensFunc(r_by_re, p->beta, p->ramda)*dr_by_re;
			T fb = p->fe*db;
			T dfb = p->fe*ddb;
			T phi_bb = p->A*this->densFunc(r_by_re, p->alpha, p->cai) - p->B*db;
			T dphi_bb = p->A*this->ddensFunc(r_by_re, p->alpha, p->cai)*dr_by_re - p->B*ddb;

			T dsum_phi = 0.5*(dfb/fa*phi_aa + dfa/fb*phi_bb + fb/fa*dphi_aa + fa/fb*dphi_bb - dfa*fb/(fa*fa)*phi_aa - fa*dfb/(fb*fb)*phi_bb);

			T dFE;

			if (rho < p->rhoin) {
				T c = rho/p->rhoin - 1.0;
				T dc = drho/p->rhoin;
				dFE = dc*(p->Fi1 + c*(2*p->Fi2 + c*(3*p->Fi3)));
			}
			else if (rho < p->rhoout)
			{
				T Fm33 = (rho < p->rhoe) ? p->Fm3 : p->Fm4;
				T c = rho/p->rhoe - 1.0;
				T dc = drho/p->rhoe;
				dFE = dc*(p->Fm1 + c*(2*p->Fm2 + c*(3*Fm33)));
			}
			else {
				T c = rho/p->rhos;
				T dc = drho/p->rhos;
				dFE = dc*(-p->Fn*p->fnn*p->fnn*std::log(c)*std::pow(c, p->fnn-1));
			}


			F = (-(0.5*dsum_phi + dFE)/r)*dir[*nn];


			//LOG_COUT << "F: " << F[0] << " " << F[1] << " " << F[2]  << " " << mi0 << std::endl;

			if (is_ghost)
			{
				#pragma omp critical
				molecules[mi0]->Fg += F;

				Fg_sum += F;
			}
			else
			{
				#pragma omp critical
				molecules[mi0]->Fn += F;

				Fn_sum += F;
			}

			++nn;
		}

		// accelleration (-1/m*dU/dx_m)
		#pragma omp critical
		{
			molecules[mi]->Fn -= Fn_sum;
			molecules[mi]->Fg -= Fg_sum;
		}

		//LOG_COUT << "a " << a[0] << " " << a[1] << " " << a[2] << std::endl;
	}


	void compute(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params)
	{
		//LOG_COUT << "compute" << std::endl;
		//LOG_COUT << "nn " << (*nn) << std::endl;

		// potential energy

		// get parameters for element (we assume all elements are the same in this model)
		const ElementParams& p = getParams(molecules[mi]->element->id);


		//LOG_COUT << "mi: " << mi << std::endl;
		
		T sum_phi = 0, rho = 0;
		auto nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			int mi0 = nmolecule_indices[*nn];

			if (mi0 < 0) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			const ElementParams& p0 = getParams(molecules[mi0]->element->id);

			T r = std::sqrt(dist2[*nn]);

			//LOG_COUT << "nn " << (*nn) << " r: " << r << std::endl;

			T r_by_re = r/p.re;
			T r_by_re0 = r/p0.re;

			// electron density
			rho += p0.fe*this->densFunc(r_by_re0, p0.beta, p0.ramda1);

			// pair potential sum

			T da = this->densFunc(r_by_re0, p0.beta, p0.ramda);
			T fa = p0.fe*da;
			T phi_aa = p0.A*this->densFunc(r_by_re0, p0.alpha, p0.cai) - p0.B*da;

			T db = this->densFunc(r_by_re, p.beta, p.ramda);
			T fb = p.fe*db;
			T phi_bb = p.A*this->densFunc(r_by_re, p.alpha, p.cai) - p.B*db;

			sum_phi += 0.5*(fb/fa*phi_aa + fa/fb*phi_bb);
			
			++nn;
		}

		ip_params[0] = rho;
		ip_params[1] = sum_phi;
		
		U_from_interpolation_params(*molecules[mi], ip_params);
	
		// accelerations

		ublas::c_vector<T, DIM> F;
		ublas::c_vector<T, DIM> Fn_sum, Fg_sum;
		std::vector<T> dip_params(2);

		std::fill(Fg_sum.begin(), Fg_sum.end(), (T)0);
		std::fill(Fn_sum.begin(), Fn_sum.end(), (T)0);

		nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			// compute potential gradient w.r.t. molecules[nn]->x

			int mi0 = nmolecule_indices[*nn];
			bool is_ghost = mi0 < 0;

			if (is_ghost) {
				mi0 = gmolecules[-1 - mi0]->molecule_index;
			}

			// get parameters for element
			const ElementParams& p0 = getParams(molecules[mi0]->element->id);

			T r = std::sqrt(dist2[*nn]);
			T r_by_re = r/p.re;
			T dr_by_re = 1.0/p.re;
			T r_by_re0 = r/p0.re;
			T dr_by_re0 = 1.0/p0.re;

			// electron density
			dip_params[0] = p0.fe*this->ddensFunc(r_by_re0, p0.beta, p0.ramda1)*dr_by_re0; // drho

			// pair potential sum

			T da = this->densFunc(r_by_re0, p0.beta, p0.ramda);
			T dda = this->ddensFunc(r_by_re0, p0.beta, p0.ramda)*dr_by_re0;
			T fa = p0.fe*da;
			T dfa = p0.fe*dda;
			T phi_aa = p0.A*this->densFunc(r_by_re0, p0.alpha, p0.cai) - p0.B*da;
			T dphi_aa = p0.A*this->ddensFunc(r_by_re0, p0.alpha, p0.cai)*dr_by_re0 - p0.B*dda;

			T db = this->densFunc(r_by_re, p.beta, p.ramda);
			T ddb = this->ddensFunc(r_by_re, p.beta, p.ramda)*dr_by_re;
			T fb = p.fe*db;
			T dfb = p.fe*ddb;
			T phi_bb = p.A*this->densFunc(r_by_re, p.alpha, p.cai) - p.B*db;
			T dphi_bb = p.A*this->ddensFunc(r_by_re, p.alpha, p.cai)*dr_by_re - p.B*ddb;

			dip_params[1] = 0.5*(dfb/fa*phi_aa + dfa/fb*phi_bb + fb/fa*dphi_aa + fa/fb*dphi_bb - dfa*fb/(fa*fa)*phi_aa - fa*dfb/(fb*fb)*phi_bb);	// dsum_phi

			T dU = dU_from_interpolation_params(*molecules[mi], ip_params, dip_params);

			F = (-(dU)/r)*dir[*nn];

			//LOG_COUT << "F: " << F[0] << " " << F[1] << " " << F[2]  << " " << mi0 << std::endl;

			if (is_ghost)
			{
				#pragma omp critical
				molecules[mi0]->Fg += F;

				Fg_sum += F;
			}
			else
			{
				#pragma omp critical
				molecules[mi0]->Fn += F;

				Fn_sum += F;
			}

			++nn;
		}

		// accelleration (-1/m*dU/dx_m)
		#pragma omp critical
		{
			molecules[mi]->Fn -= Fn_sum;
			molecules[mi]->Fg -= Fg_sum;
		}
	}

	std::size_t num_interpolation_params() const
	{
		return 2;
	}

	void U_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params) const
	{
		const ElementParams& p = getParams(m.element->id);

		// embedding energy
		T rho = params[0];
		T FE;
		if (rho < p.rhoin) {
			T c = rho/p.rhoin - 1.0;
			FE = p.Fi0 + c*(p.Fi1 + c*(p.Fi2 + c*p.Fi3));
		}
		else if (rho < p.rhoout)
		{
			T Fm33 = (rho < p.rhoe) ? p.Fm3 : p.Fm4;
			T c = rho/p.rhoe - 1.0;
			FE = p.Fm0 + c*(p.Fm1 + c*(p.Fm2 + c*Fm33));
		}
		else {
			T c = rho/p.rhos;
			FE = p.Fn*(1.0 - p.fnn*std::log(c))*std::pow(c, p.fnn);
		}

		// EAM potential
		m.U = 0.5*params[1] + FE;

		//LOG_COUT << "rho " << rho << " FE " << FE << (boost::format(" U=%g") % m.U).str()  << std::endl;
	}

	T dU_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params, const std::vector<T>& dparams) const
	{
		const ElementParams& p = getParams(m.element->id);

		// embedding energy
		T rho = params[0];
		T drho = dparams[0];
		T dFE;
		if (rho < p.rhoin) {
			T c = rho/p.rhoin - 1.0;
			T dc = drho/p.rhoin;
			dFE = dc*(p.Fi1 + c*(2*p.Fi2 + c*(3*p.Fi3)));
		}
		else if (rho < p.rhoout)
		{
			T Fm33 = (rho < p.rhoe) ? p.Fm3 : p.Fm4;
			T c = rho/p.rhoe - 1.0;
			T dc = drho/p.rhoe;
			dFE = dc*(p.Fm1 + c*(2*p.Fm2 + c*(3*Fm33)));
		}
		else {
			T c = rho/p.rhos;
			T dc = drho/p.rhos;
			dFE = dc*(-p.Fn*p.fnn*p.fnn*std::log(c)*std::pow(c, p.fnn-1));
		}

		return 0.5*dparams[1] + dFE;
	}
};



template <typename T, int DIM>
class SuttonChenPotential : public PairPotential<T, DIM>
{
protected:
	typedef struct
	{
		T a, m, n, eps, c;
	} ElementParams;

	typedef std::map<std::string, boost::shared_ptr<ElementParams> > ParamMap;
	
	ParamMap _param_map;

	void addElement(std::string element_name, boost::shared_ptr<ElementParams> params)
	{
		// the units in the file are Angstrom and eV and need to be converted to SI units
		params->a *= unit_length;
		params->eps *= unit_energy;

		_param_map.insert(typename ParamMap::value_type(element_name, params));
		LOG_COUT << "sc " << element_name << std::endl;
	}

	inline const ElementParams& getParams(std::string& id) const
	{
		typename ParamMap::const_iterator parami = _param_map.find(id);
		if (parami == _param_map.end()) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("No SC parameters for element id '%s'") % id).str()));
		}
		return *(parami->second);
	}

public:
	void readSettings(const ptree::ptree& pt)
	{
		std::string filename = "../../data/sutton_chen/parameters";
		boost::filesystem::ifstream fh(filename);
		std::string line;
		std::string element_name;
		std::size_t iparam = 0;
		std::size_t num_params = 5;
		boost::shared_ptr<ElementParams> params;

		while (std::getline(fh, line))
		{
			if (line.length() == 0)
				continue;

			if (std::isalpha(line[0])) {
				if (iparam == num_params) {
					addElement(element_name, params);
				}
				params.reset(new ElementParams());
				element_name = line;
				iparam = 0;
				continue;
			}
			
			T* params_ptr = (T*) params.get();
			params_ptr[iparam] = boost::lexical_cast<T>(line);

			iparam++;
		}

		if (iparam == num_params) {
			addElement(element_name, params);
		}
	}


	void compute_(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params)
	{
		// get parameters for element (we assume all elements are the same in this model)
		typename ParamMap::iterator parami = _param_map.find(molecules[mi]->element->id);
		if (parami == _param_map.end()) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("No SC parameters for element id '%s'") % molecules[mi]->element->id).str()));
		}
		boost::shared_ptr<ElementParams> param = parami->second;


		// potential energy

		T sum_V = 0, sum_rho = 0;
		auto nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			//LOG_COUT << "nn " << (*nn) << std::endl;

			T a_by_r = param->a/std::sqrt(dist2[*nn]);

			sum_V += std::pow(a_by_r, param->n);
			sum_rho += std::pow(a_by_r, param->m);

			++nn;
		}

		molecules[mi]->U = param->eps*(0.5*sum_V - param->c*std::sqrt(sum_rho));


		// accelerations

		ublas::c_vector<T, DIM> F;
		ublas::c_vector<T, DIM> Fn_sum, Fg_sum;

		std::fill(Fg_sum.begin(), Fg_sum.end(), (T)0);
		std::fill(Fn_sum.begin(), Fn_sum.end(), (T)0);

		nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			// compute potential gradient w.r.t. molecules[nn]->x

			T a_by_r = param->a/std::sqrt(dist2[*nn]);

			F = (0.5*param->eps*(
				param->n*std::pow(a_by_r, param->n-1)
				- param->c/std::sqrt(sum_rho)*param->m*std::pow(a_by_r, param->m-1))
			*(a_by_r/dist2[*nn]))*dir[*nn];

			int mi0 = nmolecule_indices[*nn];

			if (mi0 < 0)
			{
				mi0 = gmolecules[-1 - mi0]->molecule_index;
				
				#pragma omp critical
				molecules[mi0]->Fg += F;

				Fg_sum += F;
			}
			else
			{
				#pragma omp critical
				molecules[mi0]->Fn += F;

				Fn_sum += F;
			}

			++nn;
		}

		// accelleration (-1/m*dU/dx_m)
		#pragma omp critical
		{
			molecules[mi]->Fn -= Fn_sum;
			molecules[mi]->Fg -= Fg_sum;
		}

		//LOG_COUT << "." << std::endl;

		//LOG_COUT << "a " << a[0] << " " << a[1] << " " << a[2] << std::endl;
	}

	void compute(
		std::size_t mi,
		const std::vector<std::size_t>& nn_indices,
		const std::vector<int>& nmolecule_indices,
		const std::vector<T>& dist2,
		const std::vector< ublas::c_vector<T, DIM> >& dir,
		std::vector< boost::shared_ptr< Molecule<T, DIM> > >& molecules,
		std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > >& gmolecules,
		std::vector<T>& ip_params)
	{
		// get parameters for element (we assume all elements are the same in this model)
		const ElementParams& p = getParams(molecules[mi]->element->id);

		// potential energy

		T sum_V = 0, sum_rho = 0;
		auto nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			//LOG_COUT << "nn " << (*nn) << std::endl;

			T a_by_r = p.a/std::sqrt(dist2[*nn]);

			sum_V += std::pow(a_by_r, p.n);
			sum_rho += std::pow(a_by_r, p.m);

			++nn;
		}

		ip_params[0] = sum_V;
		ip_params[1] = sum_rho;
		U_from_interpolation_params(*molecules[mi], ip_params);

		// accelerations

		ublas::c_vector<T, DIM> F;
		ublas::c_vector<T, DIM> Fn_sum, Fg_sum;
		std::vector<T> dip_params(2);

		std::fill(Fn_sum.begin(), Fn_sum.end(), (T)0);
		std::fill(Fg_sum.begin(), Fg_sum.end(), (T)0);

		nn = nn_indices.begin();
		while (nn != nn_indices.end())
		{
			// compute potential gradient w.r.t. molecules[nn]->x

			T r = std::sqrt(dist2[*nn]);
			T a_by_r = p.a/r;
			T a_by_r2 = a_by_r/r;

			dip_params[0] = -p.n*std::pow(a_by_r, p.n-1)*a_by_r2; // dsum_V
			dip_params[1] = -p.m*std::pow(a_by_r, p.m-1)*a_by_r2; // dsum_rho

			T dU = dU_from_interpolation_params(*molecules[mi], ip_params, dip_params);

			F = (-(dU)/r)*dir[*nn];

			int mi0 = nmolecule_indices[*nn];

			if (mi0 < 0)
			{
				mi0 = gmolecules[-1 - mi0]->molecule_index;
				
				#pragma omp critical
				molecules[mi0]->Fg += F;

				Fg_sum += F;
			}
			else
			{
				#pragma omp critical
				molecules[mi0]->Fn += F;

				Fn_sum += F;
			}

			++nn;
		}

		#pragma omp critical
		{
			molecules[mi]->Fn -= Fn_sum;
			molecules[mi]->Fg -= Fg_sum;
		}
	}

	std::size_t num_interpolation_params() const
	{
		return 2;
	}

	void U_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params) const
	{
		const ElementParams& p = getParams(m.element->id);

		m.U = p.eps*(0.5*params[0] - p.c*std::sqrt(params[1]));
	}

	T dU_from_interpolation_params(Molecule<T, DIM>& m, const std::vector<T>& params, const std::vector<T>& dparams) const
	{
		const ElementParams& p = getParams(m.element->id);

		return p.eps*0.5*(dparams[0] - p.c/std::sqrt(params[1])*dparams[1]);
	}
};




template <typename T, int DIM>
class MDSolver
{
public:
	typedef boost::function<bool(float)> TimestepCallback;

protected:
	bool _zero_mean;
	bool _shuffle_elements;
	bool _periodic_bc;
	bool _periodic_wrap;
	bool _write_ghosts;
	bool _interpolate_potential;
	std::size_t _interpolate_potential_points;
	std::string _potential_type;
	std::string _cell_type;
	std::string _result_filename;
	std::string _element_db_filename;
	std::string _initial_position;	// fcc, random
	std::string _initial_velocity;	// boltzmann, uniform, zero
	std::string _time_step_mode;
	std::string _xml_project_str;
	std::string _thermostat;
	std::string _barostat;
	std::size_t _lattice_multiplier;
	std::size_t _N;
	int _seed;
	std::size_t _store_interval;
	std::size_t _callback_interval;
	std::size_t _store_number;
	boost::shared_ptr< UnitCell<T, DIM> > _cell;
	boost::shared_ptr< VerletMap<T, DIM> > _vmap;
	boost::shared_ptr< ElementDatabase<T, DIM> > _element_db;
	boost::shared_ptr< PairPotential<T, DIM> > _potential;
	std::array<std::size_t, DIM> _vdims;
	T _nn_levels;	// relative number of nearst neighbours included in the Verlet map
	T _nn_radius, _nn_radius2;
	std::string _nn_update;

	T _ds_max;
	T _tau_T;
	T _tau_p;
	T _T0_scale;

	std::vector< boost::shared_ptr< Molecule<T, DIM> > > _molecules;
	std::vector< boost::shared_ptr< GhostMolecule<T, DIM> > > _ghost_molecules;
	std::vector< boost::shared_ptr< SimulationInterval<T, DIM> > > _intervals;
	std::vector< boost::shared_ptr< ElementFraction<T, DIM> > > _fractions;

	T _current_Ekin;
	T _current_Epot;
	T _stats_Ekin;
	T _stats_Epot;
	T _stats_p;
	T _stats_V;
	ublas::c_vector<T, DIM> _stats_mv;
	T _stats_dt;
	T _stats_t;
	std::size_t _stats_interval;
	std::size_t _stats_steps;
	T _unit_V;

	// callbacks
	TimestepCallback _timestep_callback;

public:
	MDSolver()
	{
		_unit_V = std::pow(unit_length, DIM);

		_cell_type = "cubic";
		_potential_type = "embedded-atom";
		_nn_radius = -1.0;
		_nn_radius2 = 1.0;
		_nn_levels = 1.0;
		_nn_update = "always";
		_result_filename = "results.xml";
		_element_db_filename = "";
		_callback_interval = 100;
		_store_interval = 0;
		_store_number = 500;
		_periodic_bc = true;
		_periodic_wrap = true;
		_interpolate_potential = false;
		_interpolate_potential_points = 400;
		_write_ghosts = false;
		_zero_mean = true;
		_initial_position = "fcc";
		_lattice_multiplier = 1;
		_initial_velocity = "boltzmann";
		_ds_max = 0.01;
		_tau_T = 1.0*unit_time;
		_tau_p = 1.0*unit_time*unit_p;
		_time_step_mode = "fixed";
		_T0_scale = 1.0;
		_xml_project_str = "";
		_shuffle_elements = true;
		_thermostat = "berendsen";
		_barostat = "none";
	}

	inline void setTimestepCallback(TimestepCallback cb)
	{
		_timestep_callback = cb;
	}

	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		_N = pt_get(pt, "n", _N);
		_seed = pt_get(pt, "seed", _seed);

		_tau_T = pt_get(pt, "tau_T", _tau_T/unit_time)*unit_time;
		_tau_p = pt_get(pt, "tau_p", _tau_p/(unit_time*unit_p))*unit_time*unit_p;
		_ds_max = pt_get(pt, "ds_max", _ds_max/unit_length)*unit_length;
		_initial_position = pt_get(pt, "initial_position", _initial_position);
		_initial_velocity = pt_get(pt, "initial_velocity", _initial_velocity);
		_thermostat = pt_get(pt, "thermostat", _thermostat);
		_barostat = pt_get(pt, "barostat", _barostat);
		_lattice_multiplier = pt_get(pt, "lattice_multiplier", _lattice_multiplier);
		_periodic_bc = pt_get(pt, "periodic_bc", _periodic_bc);
		_periodic_wrap = pt_get(pt, "periodic_wrap", _periodic_wrap);
		_interpolate_potential = pt_get(pt, "interpolate_potential", _interpolate_potential);
		_interpolate_potential_points = pt_get(pt, "interpolate_potential_points", _interpolate_potential_points);
		_write_ghosts = pt_get(pt, "write_ghosts", _write_ghosts);
		_zero_mean = pt_get(pt, "zero_mean", _zero_mean);
		_result_filename = pt_get(pt, "result_filename", _result_filename);
		_time_step_mode = pt_get(pt, "time_step_mode", _time_step_mode);
		_store_interval = pt_get(pt, "store_interval", _store_interval);
		_store_number = pt_get(pt, "store_number", _store_number);
		_nn_levels = pt_get(pt, "nn_levels", _nn_levels);
		_nn_update = pt_get(pt, "nn_update", _nn_update);
		_nn_radius = pt_get(pt, "nn_radius", _nn_radius/unit_length)*unit_length;
		_nn_radius2 = _nn_radius*_nn_radius;
		_T0_scale = pt_get(pt, "T0_scale", _T0_scale);
		_shuffle_elements = pt_get(pt, "shuffle_elements", _shuffle_elements);

		// load element database
		_element_db_filename = pt_get(pt, "element_db", _element_db_filename);
		_element_db.reset(new ElementDatabase<T, DIM>());
		if (_element_db_filename != "") {
			_element_db->load(_element_db_filename);
		}
		const ptree::ptree& elements = pt.get_child("elements", empty_ptree);
		_element_db->load_xml(elements);

		// read the cell

		const ptree::ptree& cell = pt.get_child("cell", empty_ptree);
		const ptree::ptree& cell_attr = cell.get_child("<xmlattr>", empty_ptree);
		_cell_type = pt_get<std::string>(cell_attr, "type", _cell_type);

		if (_cell_type == "cubic") {
			_cell.reset(new CubicUnitCell<T, DIM>());
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown cell type '%s'") % _cell_type).str()));
		}

		_cell->readSettings(cell);

		// read volume fractions

		std::list< boost::shared_ptr<Element<T,DIM> > > element_list;
		const ptree::ptree& fractions = pt.get_child("fractions", empty_ptree);

		BOOST_FOREACH(const ptree::ptree::value_type &v, fractions)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "fraction") {
				boost::shared_ptr< ElementFraction<T, DIM> > fr;
				fr.reset(new ElementFraction<T, DIM>());

				std::string element_id = pt_get<std::string>(attr, "element", "");
				fr->element = _element_db->get(element_id);
				fr->fraction = pt_get<T>(attr, "value", 0.0);

				if (fr->fraction > 0.0) {
					_fractions.push_back(fr);
					element_list.push_back(fr->element);
				}
			}
		}

		// read the potential

		const ptree::ptree& potential = pt.get_child("potential", empty_ptree);
		const ptree::ptree& potential_attr = potential.get_child("<xmlattr>", empty_ptree);
		_potential_type = pt_get<std::string>(potential_attr, "type", _potential_type);

		if (_potential_type == "sutton-chen") {
			_potential.reset(new SuttonChenPotential<T, DIM>());
		}
		else if (_potential_type == "embedded-atom") {
			_potential.reset(new EmbeddedAtomPotential<T, DIM>());
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown potential type '%s'") % _potential_type).str()));
		}

		_potential->readSettings(potential);

		if (_interpolate_potential) {
			// TODO: better value for r_min?
			_potential.reset(new InterpolatedPotential<T, DIM>(_potential, 0.8*unit_length, 1.3*_nn_radius, _interpolate_potential_points, element_list));
			_potential->readSettings(potential);
		}

		// read time steps

		const ptree::ptree& steps = pt.get_child("steps", empty_ptree);

		T last_t = 0 * unit_time;
		T last_T = 300 * unit_T;
		T last_p = 1 * unit_p;
		T last_V = _cell->volume();
		T last_dt = pt_get<T>(pt, "dt", 1.0) * unit_time;

		BOOST_FOREACH(const ptree::ptree::value_type &v, steps)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "step") {
				boost::shared_ptr< SimulationInterval<T, DIM> > iv;
				iv.reset(new SimulationInterval<T, DIM>());
				iv->t = pt_get<T>(attr, "t", last_t/unit_time) * unit_time;
				iv->T = pt_get<T>(attr, "T", last_T/unit_T) * unit_T;
				iv->enforce_T = pt_get<bool>(attr, "enforce_T", false);
				iv->enforce_mean_T = pt_get<bool>(attr, "enforce_mean_T", false);
				iv->p = pt_get<T>(attr, "p", last_p/unit_p) * unit_p;
				iv->V = pt_get<T>(attr, "V", last_V/_unit_V) * _unit_V;
				iv->dt = pt_get<T>(attr, "dt", last_dt/unit_time) * unit_time;
				last_t = iv->t;
				last_T = iv->T;
				last_p = iv->p;
				last_V = iv->V;
				last_dt = iv->dt;
				_intervals.push_back(iv);
			}
		}


		// fetch the whole xml object as string
		{
			std::stringstream ss;
			std::size_t indent = 1;
			char indent_char = '\t';
	#if BOOST_VERSION < 105800
			ptree::xml_writer_settings<char> settings(indent_char, indent);
	#else
			ptree::xml_writer_settings<std::string> settings = boost::property_tree::xml_writer_make_settings<std::string>(indent_char, indent);
	#endif

			#if 0
				write_xml(ss, pt, settings);	// adds <?xml declaration
			#else
				write_xml_element(ss, std::string(), pt, -1, settings);	// omits <?xml declaration
			#endif

			_xml_project_str = ss.str();
			boost::trim(_xml_project_str);
		}
	}

	void init()
	{
		LOG_COUT << "init" << std::endl;


		// adjust element count for position patterns
		// and scale the cell

		if (_initial_position == "fcc")
		{
			_N = 4*std::pow(_lattice_multiplier, DIM);
		}
		else if (_initial_position == "bcc")
		{
			_N = 2*std::pow(_lattice_multiplier, DIM);
		}
		else if (_initial_position == "hcp")
		{
			_N = std::pow(_lattice_multiplier, DIM);
		}

		_cell->scale(_lattice_multiplier);

		// sort element fractions
		
		struct fr_greater
		{
		    inline bool operator() (const boost::shared_ptr< ElementFraction<T, DIM> >& a, const boost::shared_ptr< ElementFraction<T, DIM> >& b)
		    {
			return (a->fraction > b->fraction);
		    }
		};

		std::sort(_fractions.begin(), _fractions.end(), fr_greater());

		auto fr = _fractions.begin();
		std::size_t N = 0;
		while (fr != _fractions.end())
		{
			std::size_t n = (std::size_t) std::round((*fr)->fraction * _N);
			n = std::min(n, _N - N);
			(*fr)->n = n;
			N += n;
			++fr;
		}

		if (N < _N) {
			_fractions[0]->n += _N - N;
		}

		fr = _fractions.begin();
		while (fr != _fractions.end()) {
			LOG_COUT << "fraction " << (*fr)->element->id << " = "
				<< (*fr)->fraction << " (n=" << (*fr)->n << ")" << std::endl;
			++fr;
		}

		// compute number of Verlet divisions for each dimension
		const ublas::c_vector<T, DIM>& bb_size = _cell->bb_size();
		T vol_per_element = _cell->volume()/_N;
		
		if (_nn_radius < 0) {
			_nn_radius = std::pow(vol_per_element, 1/(T)DIM)*_nn_levels;
			_nn_radius2 = _nn_radius*_nn_radius;
		}

		for (std::size_t d = 0; d < DIM; d++) {
			_vdims[d] = std::max((int)std::floor(bb_size[d]/_nn_radius), 1) + 2;	// +2 for periodic boundary cells
		}

		LOG_COUT << "Verlet dimensions x = " << _vdims[0] << std::endl;
		LOG_COUT << "Nearst neighbour radius = " << _nn_radius << std::endl;

		_vmap.reset(new VerletMap<T, DIM>(_vdims));

		init_molecules();
	}

	void move_molecule(std::size_t id, const ublas::c_vector<T, DIM>& p)
	{
		if (0 <= id && id < _molecules.size()) {
			_molecules[id]->x = p;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Molecule id=%d not found") % id).str()));
		}
	}


	T calc_timestep(T dt0)
	{
		T dt;

		// compute timestep
		if (_time_step_mode == "ds_max") {
			dt = boost::numeric::bounds<T>::highest();
			auto mi = _molecules.begin();
			while (mi != _molecules.end()) {
				T v = std::sqrt(ublas::inner_prod((*mi)->v, (*mi)->v));
				T a = std::sqrt(ublas::inner_prod((*mi)->F, (*mi)->F)) / (*mi)->element->m;
				T dt_opt = 1/(v/_ds_max + std::sqrt(a/(2*_ds_max)));
				dt = std::min(dt, dt_opt);
				++mi;
			}
		}
		else {
			dt = dt0;
		}

		return dt;
	}

	void run()
	{
		Timer __t("run", false);

		// init forces
		compute_forces();

		// init stats
		reset_stats();

		LOG_COUT << "result file: " << _result_filename << std::endl;
		std::ofstream f;
		f.open(_result_filename);
		//std::ostream& f = std::cout;
		
		write_header(f);

		if (_intervals.size() < 2) {
			BOOST_THROW_EXCEPTION(std::runtime_error("Less than 2 simulation steps defined"));
		}

		auto it0 = _intervals.begin();
		auto it1 = std::next(it0);
		auto iE = _intervals.back();
		T tB = (*it0)->t;
		T tE = iE->t;
		std::size_t istep = 0;
		pt::ptime last_callback = pt::microsec_clock::universal_time();
		T last_store = boost::numeric::bounds<T>::lowest();
		ProgressBar<T> pb(tE - tB, 10000);

		T mean_Ekin = 0;
		T mean_Epot = 0;
		T mean_dt = 0;
		T last_Ekin, last_Epot;
		calc_Energy(last_Ekin, last_Epot);
		_current_Ekin = last_Ekin;
		_current_Epot = last_Epot;
		_stats_interval = 0;

		while (it1 != _intervals.end())
		{
			T t = (*it0)->t;
			T p = (*it0)->p;
			T V = (*it0)->V;
			T Tp = (*it0)->T;
			T t1 = (*it1)->t;
			T dt = ((*it1)->t - (*it0)->t);
			T dTdt = ((*it1)->T - (*it0)->T) / dt;
			T dpdt = ((*it1)->p - (*it0)->p) / dt;
			T dVdt = ((*it1)->V - (*it0)->V) / dt;
			bool last_interval = false;

			if (std::next(it1) == _intervals.end()) {
				last_interval = true;
			}

			if ((*it0)->enforce_mean_T) {
				mean_Ekin = 0;
				mean_Epot = 0;
				mean_dt = 0;
			}

			_stats_t = t;

			for(;;)
			{
				// compute timestep
				dt = calc_timestep((*it0)->dt);
				dt = std::min(dt, t1 - t);

				if ((*it0)->enforce_T) {
					set_temperature(Tp, _current_Ekin);
				}

				acc_stats(dt);
				mean_Ekin += _current_Ekin*dt;
				mean_Epot += _current_Epot*dt;
				mean_dt += dt;

				bool last_timestep = (t == t1);
				bool last_step = last_interval && last_timestep;

				bool do_store = std::ceil((t - tB) / (tE - tB) * _store_number) >
					std::ceil((last_store - tB) / (tE - tB) * _store_number);
				//if (istep % _store_interval == 0 || last_timestep) {
				if (_store_interval > 0 && istep % _store_interval == 0) {
					do_store = true;
				}
				if (last_timestep) {
					// this causes artefacts in the visualization (i.e. if the average values include only a few steps)
					//do_store = true;
				}
				if (_stats_dt == 0) {
					do_store = false;
				}

				if (do_store) {
					write_timestep(istep, t, Tp, p, V, f);
					last_store = t;
					if (pb.update(t - tB)) {
						T est = 0.5 + __t.seconds()*(100 - pb.progress())/(std::max((T)1, pb.progress()));
						pb.message() << " estimated runtime: " << getFormattedTime((long)est) << " at t=" << t << pb.end();
					}
				}
				
				if (!last_step) {
					perform_timestep(istep, t, dt, Tp, dTdt*dt, p, dpdt*dt, V, dVdt*dt);
					istep++;
				}

				if (_timestep_callback) {
					pt::ptime now = pt::microsec_clock::universal_time();
					bool do_callback = (now - last_callback).total_milliseconds() > (long)_callback_interval;
					if (do_callback) {
						T progress = (t - tB)/(tE - tB);
						if (_timestep_callback((float)progress)) {
							LOG_COUT << std::endl;
							LOG_COUT << "Simulation canceled." << std::endl;
							break;
						}
						last_callback = now;
					}
				}

				if (last_timestep) {
					break;
				}

				t = (dt < 0) ? std::max(t + dt, t1) : std::min(t + dt, t1);
				_stats_t = t;
				p += dpdt*dt;
				Tp += dTdt*dt;
				V += dVdt*dt;
				last_Ekin = _current_Ekin;
				last_Epot = _current_Epot;
			}

			if ((*it0)->enforce_mean_T)
			{
				T Ekin_mean = mean_Ekin/mean_dt;
				T Epot_mean = mean_Epot/mean_dt;
				
				T Tmean = Ekin_mean/(DIM/2.0*_molecules.size()*const_kB);
				LOG_COUT << "Tmean = " << (Tmean) << std::endl;
				set_temperature(std::max((T)0, 2*(*it0)->T - Tmean), Ekin_mean);

				/*
				Ekin + Epot = C
				
				T Ekin_set = Ekin_mean + Epot_mean - _current_Epot;
				T T_set = Ekin_set*2.0/(DIM*_molecules.size()*const_kB);

				LOG_COUT << "t = " << (t) << std::endl;
				LOG_COUT << "Ekin_set = " << (Ekin_set/unit_energy) << std::endl;
				LOG_COUT << "T_set = " << (T_set/unit_energy) << std::endl;
				LOG_COUT << "Ekin_mean = " << (Ekin_mean/unit_energy) << std::endl;
				LOG_COUT << "Epot_mean = " << (Epot_mean/unit_energy) << std::endl;

				set_temperature(T_set, _current_Ekin);

				T x, y;
				calc_Energy(x, y);
				LOG_COUT << "Get Ekin = " << x << std::endl;
				LOG_COUT << "Get Epot = " << y << std::endl;
				*/
			}

			++it0;
			++it1;
			++_stats_interval;
		}

		//LOG_COUT << std::endl;
		write_tailing(f);
	}

	void write_header(std::ostream & f)
	{
		f << "<results>\n";
		
		f << "\t<project>\n\t\t";
		std::string xml = _xml_project_str;
		boost::replace_all(xml, "\n", "\n\t\t");
		f << xml;
		f << "\n\t</project>\n";

		f << "\t<dim>" << DIM << "</dim>\n";
		f << "\t<nn_radius>" << (_nn_radius/unit_length) << "</nn_radius>\n";
		f << "\t<lattice_multiplier>" << _lattice_multiplier << "</lattice_multiplier>\n";
		
		f << "\t<cell>\n";
		f << "\t\t<type>" << _cell_type << "</type>\n";
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();
		f << "\t\t<size";
			for (std::size_t d = 0; d < DIM; d++)
				f << (boost::format(" a%d='%g'") % d % (L[d]/unit_length)).str();
		f << " />\n";
		f << "\t\t<origin";
			for (std::size_t d = 0; d < DIM; d++)
				f << (boost::format(" p%d='%g'") % d % (p0[d]/unit_length)).str();
		f << " />\n";
		f << "\t</cell>\n";
	
		f << "\t<elements>\n";
			auto fr = _fractions.begin();
			while (fr != _fractions.end())
			{
				f << (boost::format("\t\t<element id='%s' r='%g' fraction='%g' n='%d' color='%s' />\n") % (*fr)->element->id % ((*fr)->element->r/unit_length) % (*fr)->fraction % (*fr)->n % (*fr)->element->color).str();
				++fr;
			}
		f << "\t</elements>\n";
		
		f << "\t<molecules>\n";
			for (std::size_t i = 0; i < _molecules.size(); i++) {
				f << (boost::format("\t\t<molecule id='%d' element='%s' />\n") % i % _molecules[i]->element->id).str();
			}
		f << "\t</molecules>\n";

		f << "\t<timesteps>\n";
	}

	void write_tailing(std::ostream & f)
	{
		f << "\t</timesteps>\n";

		f << "</results>\n";
	}

	void compute_forces(bool debug = false)
	{
		Timer __t("compute_forces", false);

		// decide if nn update is required
		bool update_nn = true;
		if (_nn_update == "initial") {
			if (_molecules.size() > 0 && _molecules[0]->nn_indices.size() > 0) {
				update_nn = false;
			}
		}
		else if (_nn_update == "first_interval") {
			if (_stats_interval > 0) {
				update_nn = false;
			}
		}

		if (update_nn) {
			// only update verlet map, if necessary
			verlet_update();
		}

		// clear accelerations and store old accelerations for velocity verlet method
		auto mi = _molecules.begin();
		while (mi != _molecules.end())
		{
			boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;
			molecule->F0 = molecule->F;
			std::fill(molecule->Fn.begin(), molecule->Fn.end(), (T)0);
			std::fill(molecule->Fg.begin(), molecule->Fg.end(), (T)0);
			++mi;
		}

		std::size_t n = _vdims[0];
		for (std::size_t i = 1; i < DIM; i++) {
			n *= _vdims[i];
		}

		#pragma omp parallel for schedule(dynamic)
		for (std::size_t i = 0; i < n; i++)
		{
			bool on_boundary = false;
			std::array<std::size_t, DIM> cell_index;
			// i = i0 + i1*n0 + i2*n0*n1 + n3*n0*n1*n2 + ...
			std::size_t k = i;
			for (std::size_t j = 0; j < DIM; j++) {
				cell_index[j] = k % _vdims[j];
				k -= cell_index[j];
				k /= _vdims[j];
				on_boundary = on_boundary || (cell_index[j] == 0) || (cell_index[j] == _vdims[j]-1);
			}

			if (on_boundary) continue;

			std::vector<T> ip_params(_potential->num_interpolation_params());

			// collect all molecule indices
			boost::shared_ptr<VerletCell<T, DIM> > cell = _vmap->get_cell(cell_index);
			const std::vector<int>& molecule_indices = cell->indices();

			if (update_nn)
			{
				std::vector<int>& nmolecule_indices = cell->nindices();
				std::size_t m = std::pow(3, DIM);

				// iterate over all neighbour cells and cell itself
				nmolecule_indices.clear();
				for (std::size_t j = 0; j < m; j++)
				{
					std::size_t k = j;
					// compute cell index
					std::array<std::size_t, DIM> ncell_index;
					for (std::size_t l = 0; l < DIM; l++) {
						ncell_index[l] = k % 3;
						k -= ncell_index[l];
						k /= 3;
						ncell_index[l] += cell_index[l] - 1;
					}

					// append molecules
					boost::shared_ptr<VerletCell<T, DIM> > ncell = _vmap->get_cell(ncell_index);
					const std::vector<int>& indices = ncell->indices();
					nmolecule_indices.insert(nmolecule_indices.end(), indices.begin(), indices.end());
				}

				// loop over molecules in current cell
				std::size_t nn_count = nmolecule_indices.size();
				std::vector< ublas::c_vector<T, DIM> > dir(nn_count);
				std::vector<T> dist2(nn_count);
				std::vector<std::size_t> nn_indices;	// indices within _nn_radius
				nn_indices.reserve(nn_count);

				auto mi = molecule_indices.begin();
				while (mi != molecule_indices.end())
				{
					boost::shared_ptr< Molecule<T, DIM> > molecule = _molecules[*mi];

					nn_indices.clear();
					molecule->nn_indices.clear();

					// loop over all neighbour molecules
					for (std::size_t j = 0; j < nn_count; j++)
					{
						if (nmolecule_indices[j] < 0) {
							
							boost::shared_ptr< GhostMolecule<T, DIM> > nmolecule = _ghost_molecules[(std::size_t)(-1 - nmolecule_indices[j])];
							dir[j] = _molecules[nmolecule->molecule_index]->x + nmolecule->t - molecule->x;
						}
						else {
							// exclude molecule itself
							if (nmolecule_indices[j] == *mi) continue;

							boost::shared_ptr< Molecule<T, DIM> > nmolecule = _molecules[nmolecule_indices[j]];
							dir[j] = nmolecule->x - molecule->x;
						}

						dist2[j] = ublas::inner_prod(dir[j], dir[j]);

						if (dist2[j] <= _nn_radius2) {
							nn_indices.push_back(j);
							molecule->nn_indices.push_back(nmolecule_indices[j]);
						}
					}


					if (debug)
					{
						LOG_COUT << "molecule " << *mi << ":" << std::endl;
						for (std::size_t i = 0; i < nmolecule_indices.size(); i++) {
							LOG_COUT << " nm " << i << ": " << nmolecule_indices[i] << std::endl;
						}

						for (std::size_t i = 0; i < nn_indices.size(); i++) {
							LOG_COUT << " nn " << i << ": " << nn_indices[i] << " dist2=" << dist2[nn_indices[i]] << " dir=" << dir[nn_indices[i]][0] << std::endl;
						}
					}
					
					// compute potential energy and gradient for metallic system
					// Solhjoo_Simchi_Aashuri__Molecular_dynamics_simulation_of_melting,_solidification_and_remelting_process_of_aluminium.pdf
					// 
					// 3.2.3. The Ra2i-Tabar and Sutton (RTS) many-body potentials for the metallic FCC random alloys

					// 3.2.1. The Sutton–Chen many-body potentials for the elemental FCC metals


					// force (-dU/dx_m)

					_potential->compute(*mi, nn_indices, nmolecule_indices, dist2, dir, _molecules, _ghost_molecules, ip_params);

					++mi;
				}
			}
			else // if !update_nn
			{
				// use previous calculated nn indices

				auto mi = molecule_indices.begin();
				while (mi != molecule_indices.end())
				{
					boost::shared_ptr< Molecule<T, DIM> > molecule = _molecules[*mi];
					std::vector<int>& nmolecule_indices = molecule->nn_indices;
					std::size_t nn_count = nmolecule_indices.size();
					std::vector< ublas::c_vector<T, DIM> > dir(nn_count);
					std::vector<T> dist2(nn_count);
					std::vector<std::size_t> nn_indices(nn_count);

					// loop over all neighbour molecules
					for (std::size_t j = 0; j < nn_count; j++)
					{
						if (nmolecule_indices[j] < 0) {
							
							boost::shared_ptr< GhostMolecule<T, DIM> > nmolecule = _ghost_molecules[(std::size_t)(-1 - nmolecule_indices[j])];
							dir[j] = _molecules[nmolecule->molecule_index]->x + nmolecule->t - molecule->x;
						}
						else {
							boost::shared_ptr< Molecule<T, DIM> > nmolecule = _molecules[nmolecule_indices[j]];
							dir[j] = nmolecule->x - molecule->x;
						}

						dist2[j] = ublas::inner_prod(dir[j], dir[j]);
						nn_indices[j] = j;
					}

					_potential->compute(*mi, nn_indices, nmolecule_indices, dist2, dir, _molecules, _ghost_molecules, ip_params);

					++mi;
				}
			} // update_nn
		}

		// compute total force
		mi = _molecules.begin();
		while (mi != _molecules.end())
		{
			boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;
			molecule->F = molecule->Fn + molecule->Fg;
			++mi;
		}
	}

	void perform_timestep(std::size_t istep, T t, T dt, T Tp, T dT, T p, T dp, T _V, T _dV)
	{
		Timer __t("perform_timestep", false);

		//LOG_COUT << "perform_timestep t=" << t << " dt=" << dt << std::endl;

		{
			auto mi = _molecules.begin();
			while (mi != _molecules.end())
			{
				//LOG_COUT << "m = " << (*mi) << std::endl;

				boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;

				// velocity Verlet algorithm

				//LOG_COUT << "v=" << molecule->v[0] << " a=" << molecule->a[0] << " x=" << molecule->x[0]  << std::endl;

				molecule->x += dt*molecule->v + (0.5*dt*dt/molecule->element->m)*molecule->F;

				//LOG_COUT << "x=" << molecule->x[0] << " x=" << molecule->x[1] << " x=" << molecule->x[2]  << std::endl;
				if (_periodic_bc && _periodic_wrap) {
					// TODO: when nn_update = "never" wraping a vector causes issues
					_cell->wrap_vector(molecule->x);
				}

				++mi;
			}
		}


		bool debug = false; //(istep >= 3070) && (istep <= 3072);

		compute_forces(debug);
		

		ublas::c_vector<T, DIM> p_mean;
		std::fill(p_mean.begin(), p_mean.end(), (T)0);

		T virial = 0;
		T Ekin = _current_Ekin;

		// update velocities
		{
			auto mi = _molecules.begin();
			while (mi != _molecules.end())
			{
				boost::shared_ptr< Molecule<T, DIM> > m = *mi;

				//LOG_COUT << "#v=" << m->v[0] << " a=" << m->a[0] << " a0=" << m->a0[0] << " x=" << m->x[0]  << std::endl;

				m->v += (0.5*dt/m->element->m)*(m->F + m->F0);
				p_mean += m->element->m*m->v;

				virial += ublas::inner_prod(m->Fn, m->x);

				++mi;
			}
			
			p_mean /= (T) _molecules.size();
		}

		T Temp = Ekin/(0.5*const_kB*DIM*_molecules.size());
		T press = (2*Ekin + virial)/(DIM*_cell->volume());
		T v_scale = 1.0;
		T L_scale = 1.0;

		if (_thermostat == "berendsen")
		{
			v_scale = std::sqrt(1 + dt/_tau_T*(Tp/Temp - 1.0)); // lambda
		}

		if (_barostat == "berendsen")
		{
			// LAMMPS source: https://github.com/lammps/lammps/blob/master/src/fix_press_berendsen.cpp
			L_scale = std::pow(1 + dt/_tau_p*(p - press), (T)1.0/DIM); // mu
		}

		//T v_scale = std::exp(dT/Tp);
		//T v_scale = std::sqrt(1 + dT/Tp);

		_current_Ekin = 0;
		_current_Epot = 0;

		auto mi = _molecules.begin();
		while (mi != _molecules.end())
		{
			boost::shared_ptr< Molecule<T, DIM> > m = *mi;

			if (_zero_mean) {
				// adjust to zero mean impulse
				m->v -= p_mean/m->element->m;
			}

			// scale velocities to adjust temperature
			m->v *= v_scale;
	
			// scale positions
			m->x *= L_scale;

			// update Ekin and Epot	
			_current_Ekin += 0.5*(*mi)->element->m*ublas::inner_prod((*mi)->v, (*mi)->v);
			_current_Epot +=(*mi)->U;

			++mi;
		}

		// scale cell
		_cell->scale(L_scale);
	}

	T calc_Energy(T& Ekin, T& Epot)
	{
		Ekin = 0;
		Epot = 0;

		auto mi = _molecules.begin();
		while (mi != _molecules.end())
		{
			Ekin += 0.5*(*mi)->element->m*ublas::inner_prod((*mi)->v, (*mi)->v);
			Epot += (*mi)->U;

			++mi;
		}

		return Ekin;
	}

	void acc_stats(T dt)
	{
		T Ekin = 0, Epot = 0, p = 0;
		ublas::c_vector<T, DIM> mv;
		std::fill(mv.begin(), mv.end(), (T)0);

		for (std::size_t i = 0; i < _molecules.size(); i++) {
			Ekin += 0.5*_molecules[i]->element->m*ublas::inner_prod(_molecules[i]->v, _molecules[i]->v);
			Epot += _molecules[i]->U;
			mv += _molecules[i]->element->m*_molecules[i]->v;
			p += ublas::inner_prod(_molecules[i]->Fn, _molecules[i]->x);
		}

		T V = _cell->volume();

		p /= -V*DIM;
		p += Ekin/(DIM/2.0*V);

		_stats_Ekin += Ekin*dt;
		_stats_Epot += Epot*dt;
		_stats_p += p*dt;
		_stats_V += V*dt;
		_stats_mv += mv*dt;
		_stats_dt += dt;
		_stats_steps += 1;
	}

	void norm_stats()
	{
		if (_stats_dt != 0) {
			_stats_Ekin /= _stats_dt;
			_stats_Epot /= _stats_dt;
			_stats_p /= _stats_dt;
			_stats_V /= _stats_dt;
			_stats_mv /= _stats_dt;
		}
		_stats_dt = 1;
	}

	void reset_stats()
	{
		_stats_Ekin = 0;
		_stats_Epot = 0;
		_stats_p = 0;
		_stats_V = 0;
		std::fill(_stats_mv.begin(), _stats_mv.end(), (T)0);
		_stats_dt = 0;
		_stats_t = 0;
		_stats_steps = 0;
		_stats_interval = 0;
	}

	void write_timestep(std::size_t index, T t, T Tp, T p, T V, std::ostream & f)
	{
		Timer __t("write_timestep", false);

		f << (boost::format("\t\t<timestep id='%d' t='%g' T='%g' p='%g' V='%g'>\n") % index % (t/unit_time) % (Tp/unit_T) % (p/unit_p) % (V/_unit_V)).str();

		f << "\t\t\t<stats>\n";

			T dt = _stats_dt;
			norm_stats();

			T M = 0;
			for (std::size_t i = 0; i < _molecules.size(); i++) {
				M += _molecules[i]->element->m;
			}

			T Etot = _stats_Ekin + _stats_Epot;
			T Temp = 2.0*_stats_Ekin/(DIM*const_kB*_molecules.size());
			T mv = std::sqrt(ublas::inner_prod(_stats_mv, _stats_mv));

			f << (boost::format("<dt>%.8g</dt>") % (dt/unit_time)).str();
			f << (boost::format("<steps>%.8g</steps>") % (_stats_steps)).str();
			f << (boost::format("<Ekin>%.8g</Ekin>") % (_stats_Ekin/unit_energy)).str();
			f << (boost::format("<Epot>%.8g</Epot>") % (_stats_Epot/unit_energy)).str();
			f << (boost::format("<Etot>%.8g</Etot>") % (Etot/unit_energy)).str();
			f << (boost::format("<T>%.8g</T>") % (Temp/unit_T)).str();
			f << (boost::format("<P>%.8g</P>") % (_stats_p/unit_p)).str();
			f << (boost::format("<V>%.8g</V>") % (_stats_V/_unit_V)).str();
			f << (boost::format("<MV>%.8g</MV>") % (mv*mv/(2*M)/(unit_energy))).str();

			reset_stats();

		f << "\t\t\t</stats>\n";

		f << "\t\t\t<molecules>\n";
			for (std::size_t i = 0; i < _molecules.size(); i++) {
				f << (boost::format("\t\t\t\t<molecule id='%d'") % i).str();
				for (std::size_t d = 0; d < DIM; d++) {
					f << (boost::format(" p%d='%g'") % d % (_molecules[i]->x[d]/unit_length)).str();
					f << (boost::format(" v%d='%g'") % d % (_molecules[i]->v[d]/unit_length*unit_time)).str();
					f << (boost::format(" a%d='%g'") % d % (_molecules[i]->F[d]/_molecules[i]->element->m/unit_length*unit_time*unit_time)).str();
				}
				f << (boost::format(" U='%g'") % (_molecules[i]->U/unit_energy)).str();
				f << " />\n";
			}
		f << "\t\t\t</molecules>\n";

		if (_write_ghosts) {
			f << "\t\t\t<ghost_molecules>\n";
				auto it = _ghost_molecules.begin();
				while (it != _ghost_molecules.end()) {
					f << (boost::format("\t\t\t\t<ghost_molecule m='%d'") % (*it)->molecule_index).str();
					for (std::size_t d = 0; d < DIM; d++) {
						f << (boost::format(" t%d='%g'") % d % ((*it)->t[d]/unit_length)).str();
					}
					f << " />\n";
					++it;
				}
			f << "\t\t\t</ghost_molecules>\n";
		}

		f << "\t\t</timestep>\n";
	}

	
	void init_molecules()
	{
		Timer __t("init_molecules");

		_molecules.clear();

		RandomUniform01<T>& rndu = RandomUniform01<T>::instance();
		RandomNormal01<T>& rndn = RandomNormal01<T>::instance();
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();

		rndu.seed(_seed);
		rndn.seed(_seed);
		std::srand(_seed);

		// create N molecules based on the provided fractions
		auto fr = _fractions.begin();
		std::size_t index = 0;

		// mean velocity
		ublas::c_vector<T, DIM> p_mean;
		std::fill(p_mean.begin(), p_mean.end(), (T)0);

		while (fr != _fractions.end())
		{
			std::size_t n = (*fr)->n;

			for (std::size_t i = 0; i < n; i++)
			{
				boost::shared_ptr< Molecule<T, DIM> > m;
				m.reset(new Molecule<T, DIM>());
				m->element = (*fr)->element;

				// set initial position
				if (_initial_position == "fcc")
				{
					std::size_t sub_index = index % 4;

					std::size_t k = (index - sub_index)/4;
					// compute cell index
					std::array<std::size_t, DIM> cell_index;
					for (std::size_t l = 0; l < DIM; l++) {
						cell_index[l] = k % _lattice_multiplier;
						k -= cell_index[l];
						k /= _lattice_multiplier;
					}

					for (std::size_t j = 0; j < DIM; j++) {
						std::size_t shift = 0;
						if (sub_index > 0 && sub_index != j+1) shift = 1;
						m->x[j] = p0[j] + (4*cell_index[j] + 2*shift + 1)*L[j]/(4*_lattice_multiplier);
					}
				}
				else if (_initial_position == "hcp")
				{
					std::size_t k = index;
					// compute cell index
					std::array<std::size_t, DIM> cell_index;
					for (std::size_t l = 0; l < DIM; l++) {
						cell_index[l] = k % _lattice_multiplier;
						k -= cell_index[l];
						k /= _lattice_multiplier;
					}

					T xshift = (DIM > 1) && cell_index[1] % 2 != 0 ? (T)0.5 : (T)0;
					T xshift1 = (DIM == 1) ? (T)0.25 : (T)0;
					T yshift = (DIM > 2) && cell_index[2] % 2 != 0 ? (T)0.5 : (T)0;
					T yshift2 = (DIM == 2) ? (T)0.25 : (T)0;
					if (DIM > 0) m->x[0] = p0[0] + (cell_index[0] + 0.25 + xshift - 0.25*yshift + xshift1)*L[0]/_lattice_multiplier;
					if (DIM > 1) m->x[1] = p0[1] + (cell_index[1] + 0.25 + yshift + yshift2)*L[1]/_lattice_multiplier;
					if (DIM > 2) m->x[2] = p0[2] + (cell_index[2] + 0.25)*L[2]/_lattice_multiplier;
				}
				else if (_initial_position == "bcc")
				{
					std::size_t sub_index = index % 2;

					std::size_t k = (index - sub_index)/2;
					// compute cell index
					std::array<std::size_t, DIM> cell_index;
					for (std::size_t l = 0; l < DIM; l++) {
						cell_index[l] = k % _lattice_multiplier;
						k -= cell_index[l];
						k /= _lattice_multiplier;
					}

					for (std::size_t j = 0; j < DIM; j++) {
						m->x[j] = p0[j] + (4*cell_index[j] + 2*sub_index + 1)*L[j]/(4*_lattice_multiplier);
					}
				}
				else if (_initial_position == "random")
				{
					for (std::size_t j = 0; j < DIM; j++) {
						m->x[j] = p0[j] + L[j]*rndu.rnd();
					}
				}
				else
				{
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown initial position type '%s'") % _initial_position).str()));
				}


				//LOG_COUT << "new molecule x=" << m->x[0] << " y=" << m->x[1] << " z=" << m->x[2] << std::endl;

				if (_initial_velocity == "boltzmann")
				{
					for (std::size_t j = 0; j < DIM; j++) {
						m->v[j] = rndn.rnd();
					}
				}
				else if (_initial_velocity == "uniform")
				{
					for (std::size_t j = 0; j < DIM; j++) {
						m->v[j] = 2*rndu.rnd() - 1.0;
					}
				}
				else if (_initial_velocity == "zero")
				{
					for (std::size_t j = 0; j < DIM; j++) {
						m->v[j] = 0.0;
					}
				}
				else
				{
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown initial velocity type '%s'") % _initial_velocity).str()));
				}


				// add up mean impulse
				p_mean += m->element->m*m->v;

				// set initial acceleration to zero (to be safe)
				std::fill(m->F.begin(), m->F.end(), (T)0);

				// wrap position
				_cell->wrap_vector(m->x);

				// add molecule
				_molecules.push_back(m);

				index++;
			}

			++fr;
		}

		if (_shuffle_elements)
		{
			// shuffle elements
			std::size_t n = _molecules.size();
			for (std::size_t i = n-1; i > 0; --i) {
				std::swap(_molecules[i]->element, _molecules[std::rand() % (i+1)]->element);
			}
		}

		// adjust to zero mean impulse and compute kienetic energy

		T Ekin = 0; // kienetic enrgy

		p_mean /= (T) _molecules.size();

		{
			auto mi = _molecules.begin();
			while (mi != _molecules.end())
			{
				boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;

				(*mi)->v -= p_mean/(*mi)->element->m;

				Ekin += 0.5*(*mi)->element->m*ublas::inner_prod((*mi)->v, (*mi)->v);

				++mi;
			}
		}


		// scale velocities to match initial temperature T0

		T T0 = _T0_scale*_intervals[0]->T; // initial temperature
		set_temperature(T0, Ekin);

		/*
		{
			auto mi = _molecules.begin();
			Ekin = 0;
			while (mi != _molecules.end())
			{
				boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;

				Ekin += 0.5*(*mi)->element->m*ublas::inner_prod((*mi)->v, (*mi)->v);

				++mi;
			}

			T Temp = 2.0*Ekin/(DIM*const_kB*_molecules.size());
			LOG_COUT << "Ekin=" << Ekin << "Temp=" << Temp << std::endl;
		}
		*/

	}

	void set_temperature(T T0, T Ekin)
	{
		if (Ekin > 0)
		{
			T Ekin0 = DIM/2.0*_molecules.size()*const_kB*T0;
			T v_scale = std::sqrt(Ekin0/Ekin);

			//LOG_COUT << "Ekin0=" << Ekin0 << " Ekin=" << Ekin << " T0=" << T0 << std::endl;

			auto mi = _molecules.begin();
			while (mi != _molecules.end())
			{
				boost::shared_ptr< Molecule<T, DIM> > molecule = *mi;

				(*mi)->v *= v_scale;
				++mi;
			}

			_current_Ekin *= v_scale*v_scale;
		}
	}

	void verlet_update()
	{
		Timer __t("verlet_update", false);

		_vmap->clear();
		_ghost_molecules.clear();

		for (std::size_t i = 0; i < _molecules.size(); i++)
		{
			verlet_add(i);
			_molecules[i]->nn_indices.clear();
		}

		//_vmap->print();
	}

	// compute verlet map index from position
	// position must be within the bb of the cell
	// return true if position is valid
	bool verlet_index(const ublas::c_vector<T, DIM>& p, std::array<std::size_t, DIM>& index)
	{
		// compute verlet index
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();

		for (std::size_t i = 0; i < DIM; i++) {
			int j = (int) std::floor(1 + (p[i] - p0[i]) / L[i] * (_vdims[i] - 2));
			if (j < 0 || j >= (int)_vdims[i]) return false;
			index[i] = (std::size_t) j;
		}

		return true;
	}


	void verlet_add(std::size_t m_index)
	{
		// compute verlet index
		boost::shared_ptr< Molecule<T, DIM> > m = _molecules[m_index];
		std::array<std::size_t, DIM> index;
		verlet_index(m->x, index);
		for (std::size_t i = 0; i < DIM; i++) {
			index[i] = std::max((std::size_t) 1, std::min(index[i], _vdims[i] - 2));
		}

		// add molecule to verlet cell
		boost::shared_ptr<VerletCell<T, DIM> > v_cell = _vmap->get_cell(index);
		v_cell->add(m_index);

		if (!_periodic_bc) return;

		// add ghost elements
		const std::vector< ublas::c_vector<T, DIM> >& neighbour_translations = _cell->neighbour_translations();
		ublas::c_vector<T, DIM> p;

		for (std::size_t j = 0; j < neighbour_translations.size(); j++)
		{
			p = m->x + neighbour_translations[j];

			if (verlet_index(p, index))
			{
				// add ghost molecule to list
				int gm_index = -(int)(_ghost_molecules.size() + 1);
				boost::shared_ptr< GhostMolecule<T, DIM> > gm;
				gm.reset(new GhostMolecule<T, DIM>());
				gm->t = neighbour_translations[j];
				gm->molecule_index = m_index;
				_ghost_molecules.push_back(gm);

				// add ghost molecule to verlet cell
				boost::shared_ptr<VerletCell<T, DIM> > v_cell = _vmap->get_cell(index);
				v_cell->add(gm_index);
			}
		}

	}
};




/*

RVE generator object:
 - manage list of atoms
 - ghost atoms for periodic bc
 - generate random positions + velocities with certain distribution

Next neighbour object:
 - abstract
 - build method to initialize and update
 - query neighbour atoms for an atom up to certain configured distance

Time step object:
 - abstract
 - implements Verlet variant
 - performs time step and updates atoms positions, velocities, etc

Data storage object:
 - abstract
 - stores for each timestep atom positions, velocity, time, temperature, pressure, RVE size
 - possible implementations: memory, file, etc.

Force calculation object:
 - contains details for computing the atom acceleration

External bath coupling object:
 - abstract
 - method to compute scaling factors for positions, RVE size and velocities

MD simulation object:
 - init element database
 - init the RVE
 - init time step object
 - init force calc object
 - init data object
 - run the simulation
 - stores data to data object

configuration variables:
 - action: run, load run from file if config file has same modifiction timestamp and data was saved to file

Python export:
 - return list of elements with properties
 - return list of atoms used in simulation
 - return time series data of positions etc.
 - return evaluations (atom density vs distance plot)

GUI TODO:
 - get data
 - time slider
 - visualize atoms
 - radius scale slider
 - play speed scale slider
 - show/hide ghost atoms checkbox
 - play, pause, reset button
 - possible clickable atoms to get info of position, velocity etc.
 - buttons/tabs for diagrams and other evaluations
*/



	/*
	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		_N = pt_get<std::size_t>(pt, "n", _N);
		_V = pt_get<T>(pt, "v", _V);
		_M = pt_get<std::size_t>(pt, "m", _M);
		_seed = pt_get(pt, "seed", _seed);
		_mcs = pt_get(pt, "mcs", _mcs);
		_L = pt_get<T>(pt, "length", _L);
		_R = pt_get<T>(pt, "radius", _R);
		_dmin = pt_get<T>(pt, "dmin", _dmin);
		_dmax = pt_get<T>(pt, "dmax", _dmax);
		read_vector(pt, _dim, "dx", "dy", "dz", _dim(0), _dim(1), _dim(2));
		read_vector(pt, _x0, "x0", "y0", "z0", _x0(0), _x0(1), _x0(2));
		_periodic = pt_get(pt, "periodic", _periodic);
		//_periodic_fast = pt_get(pt, "periodic.<xmlattr>.fast", _periodic_fast);
		_planar_x = pt_get(pt, "planar.<xmlattr>.x", _planar_x);
		_planar_y = pt_get(pt, "planar.<xmlattr>.y", _planar_y);
		_planar_z = pt_get(pt, "planar.<xmlattr>.z", _planar_z);
		_periodic_x = pt_get(pt, "periodic.<xmlattr>.x", _periodic_x) && _periodic && !_planar_x;
		_periodic_y = pt_get(pt, "periodic.<xmlattr>.y", _periodic_y) && _periodic && !_planar_y;
		_periodic_z = pt_get(pt, "periodic.<xmlattr>.z", _periodic_z) && _periodic && !_planar_z;
		_intersecting = pt_get(pt, "intersecting", _intersecting);
		_type = pt_get<std::string>(pt, "type", _type);



*/



//! heamd interface (e.g. for interfacing with Python)
class HMI
{
public:
	typedef boost::function<bool(float)> TimestepCallback;

	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~HMI() {}

	//! reset solver
	virtual void reset() = 0;

	//! run actions in confing path
	virtual int run(const std::string& actions_path) = 0;

	//! cancel a running solver
	virtual void cancel() = 0;

	//! init solver
	virtual void init() = 0;

	//! set a timestep callback routine (run each timestep)
	virtual void set_timestep_callback(TimestepCallback cb) = 0;

	//! set Python heamd instance, which can be used (in Python scripts) within a project file as "hm"
	virtual void set_pyhm_instance(PyObject* instance) = 0;

	//! set Python variable, which can be used (in Python scripts/expressions) within a project file
	virtual void set_variable(std::string key, py::object value) = 0;

	//! get Python variable
	virtual py::object get_variable(std::string key) = 0;
};




//! Basic implementation of the heamd interface
template <typename T, typename R, int DIM>
class HM : public HMI
{
protected:
	boost::shared_ptr< ptree::ptree > xml_root;
	boost::shared_ptr< MDSolver<T, DIM> > solver;
	TimestepCallback timestep_callback;
	PyObject* pyhm_instance;

public:
	HM(boost::shared_ptr< ptree::ptree > xml) : xml_root(xml)
	{
		pyhm_instance = NULL;
		reset();
	}

	void set_pyhm_instance(PyObject* instance)
	{
		pyhm_instance = instance;
	}

	void set_variable(std::string key, py::object value)
	{
		PY::instance().add_local(key, value);
	}

	py::object get_variable(std::string key)
	{
		return PY::instance().get_local(key);
	}

	noinline void init_python()
	{
		const ptree::ptree& pt = xml_root->get_child("settings", empty_ptree);
		const ptree::ptree& variables = pt.get_child("variables", empty_ptree);

		// TODO: when to clear locals?
		// commented because set_variable not workin otherwise
		//PY::instance().clear_locals();

		if (pyhm_instance != NULL) {
			py::object hm(py::handle<>(py::borrowed(pyhm_instance)));
			set_variable("hm", hm);
		}

		// set variables
		BOOST_FOREACH(const ptree::ptree::value_type &v, variables)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			std::string type = pt_get<std::string>(attr, "type", "object");
			std::string value = pt_get<std::string>(attr, "value", "");
			py::object py_value;

			if (type == "str") {
				py_value = py::object(value);
			}
			else if (type == "int") {
				py_value = py::object(pt_get<long>(attr, "value"));
			}
			else if (type == "float") {
				py_value = py::object(pt_get<double>(attr, "value"));
			}
			else if (type == "object") {
				py_value = pt_get_obj(attr, "value");
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown variable type '%s' for %s") % type % v.first).str()));
			}
			
			set_variable(v.first, py_value);

			LOG_COUT << "Variable " << v.first << " = " << value << std::endl;
		}

		// execute all python blocks
		BOOST_FOREACH(const ptree::ptree::value_type &v, pt)
		{
			if (v.first == "python") {
				Timer __t(v.first);
				PY::instance().exec(v.second.data());
			}
		}
	}

	void reset()
	{
		Timer::reset_stats();
		_except.reset();

		solver.reset(new MDSolver<T, DIM>());
	}

	bool timestep_callback_wrap(float progress)
	{
		if (!timestep_callback) {
			return false;
		}

		return timestep_callback(progress);
	}

	void set_timestep_callback(TimestepCallback cb)
	{
		timestep_callback = cb;
	}

	void init() 
	{
	}

	template <class ME>
	inline std::vector<std::vector<double> > c_matrix_to_vector(const ublas::matrix_expression<ME>& m)
	{
		std::vector<std::vector<double> > v;
		for (std::size_t i = 0; i < m().size1(); i++) {
			std::vector<double> row;
			for (std::size_t j = 0; j < m().size2(); j++) {
				row.push_back(m()(i,j));
			}
			v.push_back(row);
		}
		return v;
	}

	void cancel()
	{
		set_exception("heamd canceled");
	}

	int run(const std::string& actions_path)
	{
		reset();

		// init python variables
		init_python();
		struct AfterReturn { ~AfterReturn() {
			if (PY::has_instance()) {
				//bool py_enabled = PY::instance().get_enabled();
				//PY::release();
				// recreate instance
				//PY::instance().set_enabled(py_enabled);
				//PY::instance().clear_locals();
				PY::instance().remove_local("hm");
			}
		} } ar;

		const ptree::ptree& settings = xml_root->get_child("settings", empty_ptree);

		// set print precision
		int print_precision = pt_get(settings, "print_precision", 4);
		LOG_COUT.precision(print_precision);
		std::cerr.precision(print_precision);

		// init the solver
		solver->readSettings(settings);
		solver->init();
		solver->setTimestepCallback(boost::bind(&HM<T,R,DIM>::timestep_callback_wrap, this, boost::placeholders::_1));

		int max_threads = boost::thread::hardware_concurrency();
		//int max_threads = omp_get_max_threads();

		// init OpenMP threads
		int num_threads_omp = pt_get(settings, "num_threads", 1);
		int dynamic_threads_omp = pt_get(settings, "dynamic_threads", 1);

		if (num_threads_omp < 1) {
			num_threads_omp = std::max(1, max_threads + num_threads_omp);
		}
		
		if (num_threads_omp > max_threads) {
#if 0
			BOOST_THROW_EXCEPTION(std::runtime_error(((boost::format("number of threads %d is above system limit of %d threads") % num_threads_omp) % max_threads).str()));
#else
			num_threads_omp = max_threads;
#endif
		}

		omp_set_nested(false);
		omp_set_dynamic(dynamic_threads_omp != 0);
		omp_set_num_threads(num_threads_omp);

		// Init FFTW threads
		int num_threads_fft = pt_get(settings, "fft_threads", (int)-1);

		if (num_threads_fft < 1) {
			num_threads_fft = num_threads_omp;
		}
		if (num_threads_fft > max_threads) {
			BOOST_THROW_EXCEPTION(std::runtime_error(((boost::format("number of threads %d is above system limit of %d threads") % num_threads_fft) % max_threads).str()));
		}
		if (fftw_init_threads() == 0) {
			LOG_CWARN << "could not initialize FFTW threads!" << std::endl;
		}
		fftw_plan_with_nthreads(num_threads_fft);
		
		std::string host = boost::asio::ip::host_name();
		std::string fft_wisdom = pt_get(settings, "fft_wisdom", std::string(getenv("HOME")) + "/.heamd_fft_wisdom_" + host);
		if (!fft_wisdom.empty()) {
			fftw_import_wisdom_from_filename(fft_wisdom.c_str());
		}

		LOG_COUT << "Current host: " << host << std::endl;
		LOG_COUT << "Current path: " << boost::filesystem::current_path() << std::endl;
		LOG_COUT << "FFTW wisdom: " << fft_wisdom << std::endl;
		LOG_COUT << "Running with dim=" << DIM <<
			", type=" << typeid(T).name() <<
			", result_type=" <<  typeid(R).name() <<
			", num_threads=" << num_threads_omp <<
			", fft_threads=" << num_threads_fft <<
			", max_threads=" << max_threads <<
#ifdef USE_MANY_FFT
			", many_fft" <<
#endif
			std::endl;

		LOG_COUT << "numeric bounds:" <<
			" smallest=" << boost::numeric::bounds<T>::smallest() <<
			" lowest=" << boost::numeric::bounds<T>::lowest() <<
			" highest=" << boost::numeric::bounds<T>::highest() <<
			" eps=" << std::numeric_limits<T>::epsilon() <<
			std::endl;
	
		// perform actions
		int ret = run_actions(settings, actions_path);

		// save fft wisdom
		if (!fft_wisdom.empty()) {
			fftw_export_wisdom_to_filename(fft_wisdom.c_str());
		}

		return ret;
	}

	noinline int run_actions(const ptree::ptree& settings, const std::string& path)
	{
		const ptree::ptree& actions = settings.get_child(path, empty_ptree);
		const ptree::ptree& actions_attr = actions.get_child("<xmlattr>", empty_ptree);

		if (pt_get(actions_attr, "skip", 0) != 0) {
			LOG_COUT << "skipping action: " << path << std::endl;
			return 0;
		}

		BOOST_FOREACH(const ptree::ptree::value_type &v, actions)
		{
			// check if last action failed
			if (_except) {
				return EXIT_FAILURE;
			}

			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "skip" || pt_get(attr, "skip", false)) {
				continue;
			}
			
			Timer __t(v.first, true, false);

			if (boost::starts_with(v.first, "group-"))
			{
				int ret = run_actions(settings, path + "." + v.first);
				if (ret != 0) return ret;
				continue;
			}

			if (v.first == "run")
			{
				solver->run();
			}
			else if (v.first == "move_molecule")
			{
				ublas::c_vector<T, DIM> p;
				for (int d = 0; d < DIM; d++) {
					p[d] = pt_get<T>(attr, (boost::format("p%s") % d).str(), p[d]/unit_length) * unit_length;
				}
				std::size_t id = pt_get<std::size_t>(attr, "id");
				solver->move_molecule(id, p);
			}
			else if (v.first == "print_timings")
			{
				Timer::print_stats();
			}
			else if (v.first == "exit")
			{
				return pt_get<int>(attr, "code", EXIT_SUCCESS);
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown action: '%s'") % v.first).str()));
			}
		}
		
		return EXIT_SUCCESS;
	}
};


//! Handles stuff before exit of the application
void exit_handler()
{
	LOG_COUT << DEFAULT_TEXT;

	// shut down FFTW
	fftw_cleanup_threads();
	fftw_cleanup();
}


//! Handles signals of the application
void signal_handler(int signum)
{
	LOG_COUT << std::endl;
	LOG_CERR << (boost::format("Program aborted: Signal %d received") % signum).str() << std::endl;
	print_stacktrace(LOG_CERR);
	exit(signum);  
}



//! Class which interfaces the HM interface with a project xml file
class HMProject
{
protected:
	boost::shared_ptr< ptree::ptree > xml_root;
	boost::shared_ptr< HMI > hmi;
	int xml_precision;
	PyObject* pyhm_instance;

public:
	HMProject()
	{
		this->xml_precision = -1;

		// register signal SIGINT handler and exit handler
		atexit(exit_handler);
		signal(SIGINT, signal_handler);
		signal(SIGSEGV, signal_handler);

		reset();
	}

	// reset to initial state
	void reset()
	{
		xml_root.reset(new ptree::ptree());
		hmi.reset();
		pyhm_instance = NULL;
	}

	void init_hmi()
	{
		ptree::ptree& settings = xml_root->get_child("settings", empty_ptree);

		// get type and dimension
		std::size_t dim = pt_get(settings, "dim", 3);
		std::string type = pt_get<std::string>(settings, "datatype", "double");
		std::string rtype = pt_get<std::string>(settings, "restype", "float");

	#define RUN_TYPE_AND_DIM(T, R, DIM) \
		else if (#T == type && #R == rtype && DIM == dim) hmi.reset(new HM<T, R, DIM>(xml_root))

		if (false) {}
		//RUN_TYPE_AND_DIM(double, double, 2);
		//RUN_TYPE_AND_DIM(double, double, 3);
		RUN_TYPE_AND_DIM(double, float, 1);
		RUN_TYPE_AND_DIM(double, float, 2);
		RUN_TYPE_AND_DIM(double, float, 3);
#ifdef FFTWF_ENABLED
		//RUN_TYPE_AND_DIM(float, float, 2);
		//RUN_TYPE_AND_DIM(float, float, 3);
#endif
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("dimension/datatype not supported"));
		}

		hmi->set_pyhm_instance(pyhm_instance);
	}

	boost::shared_ptr< HMI > hm()
	{
		if (!hmi) init_hmi();
		return hmi;
	}

	int run() { return hm()->run("actions"); }
	int run_path(const std::string& actions_path) { return hm()->run(actions_path); }
	void cancel() { return hm()->cancel(); }
	void init() { hm()->init(); }
	void set_pyhm_instance(py::object instance) { pyhm_instance = instance.ptr(); }
	void set_variable(std::string key, py::object value) { hm()->set_variable(key, value); }
	py::object get_variable(std::string key) { return hm()->get_variable(key); }

	/// Enable/Disable Python evaluation of expressions
	void set_py_enabled(bool enabled)
	{
		PY::instance().set_enabled(enabled);
	}

	std::string get_xml()
	{
		std::stringstream ss;
		std::size_t indent = 1;
		char indent_char = '\t';
#if BOOST_VERSION < 105800
		ptree::xml_writer_settings<char> settings(indent_char, indent);
#else
		ptree::xml_writer_settings<std::string> settings = boost::property_tree::xml_writer_make_settings<std::string>(indent_char, indent);
#endif

		#if 1
			write_xml(ss, *xml_root, settings);	// adds <?xml declaration
		#else
			write_xml_element(ss, std::string(), *xml_root, -1, settings);	// omits <?xml declaration
		#endif

		return ss.str();
	}

	typedef boost::property_tree::basic_ptree<std::basic_string<char>, std::basic_string<char> > treetype;

	ptree::ptree* get_path(const std::string& path, int create = 1)
	{
		std::string full_path = "settings." + path;
		boost::replace_all(full_path, "..", ".<xmlattr>.");

		std::vector<std::string> parts;
		boost::split(parts, full_path, boost::is_any_of("."));
		treetype* current = xml_root.get();

		for (std::size_t i = 0; i < parts.size(); i++)
		{
			std::vector<std::string> elements;
			boost::split(elements, parts[i], boost::is_any_of("[]()"));

			std::string name = elements[0];
			std::size_t index = 0;

			if (elements.size() > 1) {
				index = boost::lexical_cast<std::size_t>(elements[1]);
			}

			// find the element with the same name and index
			treetype* next = NULL;
			std::size_t counter = 0;
			for(treetype::iterator iter = current->begin(); iter != current->end(); ++iter) {
			// BOOST_FOREACH(ptree::ptree::value_type &v, *current) {
				if (iter->first == elements[0]) {
					if (counter == index) {
						if (create < 0 && i == (parts.size()-1)) {
							// delete item
							current->erase(iter);
							return NULL;
						}
						next = &(iter->second);
						break;
					}
					counter++;
				}
			}

			if (next == NULL) {
				if (create > 0) {
					// add proper number of elements
					for (std::size_t c = counter; c <= index; c++) {
						treetype::iterator v = current->push_back(treetype::value_type(name, empty_ptree));
						next = &(v->second);
					}
				}
				else if (create < 0) {
					// nothing to remove
					return NULL;
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("XML path '%s' not found") % path).str()));
				}
			}

			current = next;
		}

		return current;
	}
	
	std::string get(const std::string& key)
	{
		return this->get_path(key, 0)->get_value<std::string>("");
	}

	void erase(const std::string& key)
	{
		this->get_path(key, -1);
	}

	void set(const std::string& key)
	{
		this->get_path(key)->put_value("");
	}

	void set(const std::string& key, long value)
	{
		this->get_path(key)->put_value(value);
	}

	void set(const std::string& key, double value)
	{
		if (this->xml_precision >= 0) {
#if 1
			this->set(key, (boost::format("%g") % boost::io::group(std::setprecision(this->xml_precision), value)).str());
			return;
#else
			int exponent;
			double mantissa = std::frexp(value, exponent);
			double scale = std::pow(10.0, this->xml_precision);
			mantissa = std::round(mantissa*scale)/scale;
			value = std::ldexp(mantissa, exponent);
#endif
		}

		this->get_path(key)->put_value(value);
	}

	void set(const std::string& key, const std::string& value)
	{
		this->get_path(key)->put_value(value);
	}

	void set_xml(const std::string& xml)
	{
		std::stringstream ss;
		ss << xml;
		// read settings
		xml_root.reset(new ptree::ptree());
		read_xml(ss, *xml_root, 0*ptree::xml_parser::trim_whitespace);
	}

	void set_log_file(const std::string& logfile)
	{
		Logger::instance().setTeeFilename(logfile);
	}

	void set_xml_precision(int digits)
	{
		this->xml_precision = digits;
	}

	int get_xml_precision()
	{
		return this->xml_precision;
	}

	void load_xml(const std::string& filename)
	{
		// read settings
		xml_root.reset(new ptree::ptree());
		read_xml(filename, *xml_root, 0*ptree::xml_parser::trim_whitespace);
	}

	noinline int exec(const po::variables_map& vm)
	{
		Timer __t("application");

		std::string filename = vm["input-file"].as<std::string>();
		std::string actions_path = vm["actions-path"].as<std::string>();

		// read settings
		load_xml(filename);

		return run_path(actions_path);
	}
};


//! Python interface class for heamd
class PyHM : public HMProject
{
protected:
	py::object _py_timestep_callback;

public:

	PyHM() { }
	PyHM(py::tuple args, py::dict kw) { }

	~PyHM()
	{
		//LOG_COUT << "~PyHM" << std::endl;
		PY::release();
	}

	bool timestep_callback(float progress)
	{
		if (_py_timestep_callback) {
			py::object ret = _py_timestep_callback(progress);
			py::extract<bool> bool_ret(ret);
			if (bool_ret.check()) {
				return bool_ret();
			}
		}

		return false;
	}

	void set_timestep_callback(py::object cb)
	{
		_py_timestep_callback = cb;
		hm()->set_timestep_callback(boost::bind(&PyHM::timestep_callback, this, boost::placeholders::_1));
	}
};


py::object SetParameters(py::tuple args, py::dict kwargs)
{
	PyHM& self = py::extract<PyHM&>(args[0]);
	std::string key = py::extract<std::string>(args[1]);
	py::list keys = kwargs.keys();
	int nargs = py::len(args) + py::len(kwargs);

	if (nargs == 2) {
		self.set(key);
	}

	for(int i = 2; i < nargs; ++i)
	{
		std::string attr_key;
		py::object curArg;

		if (i < py::len(args)) {
			curArg = args[i];
			attr_key = key;
		}
		else {
			int j = i - py::len(args);
			std::string key_j = py::extract<std::string>(keys[j]);
			curArg = kwargs[keys[j]];
			attr_key = key + "." + key_j;
		}

		py::extract<long> int_arg(curArg);
		if (int_arg.check()) {
			self.set(attr_key, int_arg());
			continue;
		}
		py::extract<double> double_arg(curArg);
		if (double_arg.check()) {
			self.set(attr_key, double_arg());
			continue;
		}
		py::extract<std::string> string_arg(curArg);
		if (string_arg.check()) {
			self.set(attr_key, string_arg());
			continue;
		}

		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("invalid argument for attribute '%s' specified") % attr_key).str()));
	}

	return py::object();
}


void ExtractRangeArg(const std::string& name, size_t n, std::vector<size_t>& range, py::dict& kwargs)
{
	py::object range_arg = kwargs.get(name, py::list());
	py::list range_list = py::extract<py::list>(range_arg);
	size_t range_list_len = py::len(range_list);
	
	if (range_list_len > 0) {
		range.resize(range_list_len);
		for (size_t i = 0; i < range_list_len; i++) {
			range[i] = py::extract<size_t>(range_list[i]);
			if (range[i] < 0 || range[i] >= n) {
				BOOST_THROW_EXCEPTION(std::out_of_range("index out of range"));
			}
		}
		std::sort(range.begin(), range.end()); 
		std::vector<size_t>::iterator last = std::unique(range.begin(), range.end());
		range.erase(last, range.end());
	}
	else {
		range.resize(n);
		for (size_t i = 0; i < n; i++) {
			range[i] = i;
		}
	}
}


/*
py::object GetField(py::tuple args, py::dict kwargs)
{
	PyHM& self = py::extract<PyHM&>(args[0]);
	std::string field = py::extract<std::string>(args[1]);

	std::vector<void*> components;
	size_t nx, ny, nz, nzp, elsize;
	void* handle = self.hm()->get_raw_field(field, components, nx, ny, nz, nzp, elsize);

	std::vector<size_t> range_x, range_y, range_z, comp_range;
	ExtractRangeArg("range_x", nx, range_x, kwargs);
	ExtractRangeArg("range_y", ny, range_y, kwargs);
	ExtractRangeArg("range_z", nz, range_z, kwargs);
	ExtractRangeArg("components", components.size(), comp_range, kwargs);

#if NPY_API_VERSION >= 0x0000000B
	#define NPY_DIM_TYPE npy_intp
#else
	#define NPY_DIM_TYPE int
#endif

	std::vector<NPY_DIM_TYPE> dims(4);
	dims[0] = (NPY_DIM_TYPE) comp_range.size();
	dims[1] = (NPY_DIM_TYPE) range_x.size();
	dims[2] = (NPY_DIM_TYPE) range_y.size();
	dims[3] = (NPY_DIM_TYPE) range_z.size();

	int dtype;
	switch (elsize) {
		case 4: dtype = NPY_FLOAT; break;
		case 8: dtype = NPY_DOUBLE; break;
		default:
			BOOST_THROW_EXCEPTION(std::runtime_error("datatype not supported"));
	}

#if NPY_API_VERSION >= 0x0000000B
	py::object obj(py::handle<>(PyArray_SimpleNew(dims.size(), &dims[0], dtype)));
#else
	py::object obj(py::handle<>(PyArray_FromDims(dims.size(), &dims[0], dtype)));
#endif
	void* array_data = PyArray_DATA((PyArrayObject*) obj.ptr());

	if (PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) != (int)elsize) {
		BOOST_THROW_EXCEPTION(std::runtime_error("problem here"));
	}

	std::vector<size_t> iz_end_vec(range_z.size());
	for (std::size_t iz = 0; iz < range_z.size();) {
		size_t k = range_z[iz];
		size_t iz_end = iz + 1;
		while (iz_end < range_z.size() && range_z[iz_end] == k) { iz_end++; }
		iz_end_vec[iz] = iz_end;
		iz = iz_end;
	}

	#pragma omp parallel for schedule (static) collapse(2)
	for (size_t ic = 0; ic < comp_range.size(); ic++) {
		for (size_t ix = 0; ix < range_x.size(); ix++) {
			size_t c = comp_range[ic];
			size_t i = range_x[ix];
			for (std::size_t iy = 0; iy < range_y.size(); iy++) {
				size_t j = range_y[iy];
				size_t dest0 = ((size_t) array_data) + ((ic*range_x.size() + ix)*range_y.size() + iy)*range_z.size()*elsize;
				size_t src0 = ((size_t) components[c]) + (i*ny + j)*nzp*elsize;
				for (std::size_t iz = 0; iz < range_z.size();) {
					size_t k = range_z[iz];
					size_t iz_end = iz_end_vec[iz];
					void* dest = (void*)(dest0 + iz*elsize);
					void* src = (void*)(src0 + k*elsize);
					memcpy(dest, src, elsize*(iz_end-iz));
					iz = iz_end;
				}
			}
		}
	}

	self.hm()->free_raw_field(handle);

	return obj;
}
*/


//! Convert a vector to Python list
template<class T>
struct VecToList
{
	static PyObject* convert(const std::vector<T>& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			(*l).append(vec[i]);
		}
		return l->ptr();
	}
};


//! Convert a nested vector (rank 2 tensor) to Python list
template<class T>
struct VecVecToList
{
	static PyObject* convert(const std::vector<std::vector<T> >& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			py::list* l2 = new py::list();
			for(std::size_t j = 0; j < vec[i].size(); j++) {
				l2->append(vec[i][j]);
			}
			(*l).append(*l2);
		}
		return l->ptr();
	}
};


//! Convert a nested vector (rank 4 tensor) to Python list
template<class T>
struct VecVecVecVecToList
{
	static PyObject* convert(const std::vector<std::vector< std::vector<std::vector<T> > > >& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			py::list* l2 = new py::list();
			for(std::size_t j = 0; j < vec[i].size(); j++) {
				py::list* l3 = new py::list();
				for(std::size_t m = 0; m < vec[i][j].size(); m++) {
					py::list* l4 = new py::list();
					for(std::size_t n = 0; n < vec[i][j][m].size(); n++) {
						l4->append(vec[i][j][m][n]);
					}
					(*l3).append(*l4);
				}
				(*l2).append(*l3);
			}
			(*l).append(*l2);
		}
		return l->ptr();
	}
};


void translate1(boost::exception const& e)
{
	// Use the Python 'C' API to set up an exception object
	PyErr_SetString(PyExc_RuntimeError, boost::diagnostic_information(e).c_str());
}


void translate2(std::runtime_error const& e)
{
	// Use the Python 'C' API to set up an exception object
	PyErr_SetString(PyExc_RuntimeError, e.what());
}


void translate3(py::error_already_set const& e)
{
	// Use the Python 'C' API to set up an exception object
	//PyErr_SetString(PyExc_RuntimeError, "There was a Python error inside heamd!");
}


#if PY_VERSION_HEX >= 0x03000000
void* init_numpy() { import_array(); return NULL; }
#else
void init_numpy() { import_array(); }
#endif


py::object PyHMInitWrapper(py::tuple args, py::dict kw)
{
	py::object pyhm = args[0];
	py::object ret = pyhm.attr("__init__")(args, kw);
	PyHM& hm = py::extract<PyHM&>(pyhm);
	hm.set_pyhm_instance(pyhm);
	hm.set_py_enabled(true);
	return ret;
}


//! Python module for heamd
class PyHMModule
{
public:
	py::object HM;

	PyHMModule()
	{
		// this is required to return py::numeric::array as numpy array
		init_numpy();

		#if BOOST_VERSION < 106500
		py::numeric::array::set_module_and_type("numpy", "ndarray");
		#endif

		py::register_exception_translator<boost::exception>(&translate1);
		py::register_exception_translator<std::runtime_error>(&translate2);
		py::register_exception_translator<py::error_already_set>(&translate3);

		py::to_python_converter<std::vector<std::string>, VecToList<std::string> >();
		py::to_python_converter<std::vector<double>, VecToList<double> >();
		py::to_python_converter<std::vector<std::vector<double> >, VecVecToList<double> >();
		py::to_python_converter<std::vector<std::vector<std::vector<std::vector<double> > > >, VecVecVecVecToList<double> >();

		void (PyHM::*PyHM_set_string)(const std::string& key, const std::string& value) = &PyHM::set;
		void (PyHM::*PyHM_set_double)(const std::string& key, double value) = &PyHM::set;
		void (PyHM::*PyHM_set_int)(const std::string& key, long value) = &PyHM::set;
		void (PyHM::*PyHM_set)(const std::string& key) = &PyHM::set;

		this->HM = py::class_<PyHM, boost::shared_ptr<PyHM>, boost::noncopyable>("HM", "The heamd solver class", py::no_init)
			.def("__init__", py::raw_function(&PyHMInitWrapper), "Constructor")	// raw constructor
			.def(py::init<py::tuple, py::dict>()) // C++ constructor, shadowed by raw constructor
			.def("init", &PyHM::init, "Initialize the solver (this is usually done automatically)", py::args("self"))
			.def("run", &PyHM::run, "Runs the solver (i.e. the actions in the <actions> section)", py::args("self"))
			.def("run", &PyHM::run_path, "Run actions from a specified path in the XML tree", py::args("self", "path"))
			.def("cancel", &PyHM::cancel, "Cancel a running solver. This can be called in a callback routine for instance.", py::args("self"))
			.def("reset", &PyHM::reset, "Resets the solver to its initial state and unloads any loaded XML file.", py::args("self"))
			.def("get_xml", &PyHM::get_xml, "Get the current project configuration as XML string", py::args("self"))
			.def("set_xml", &PyHM::set_xml, "Load the current project configuration from a XML string", py::args("self"))
			.def("set_xml_precision", &PyHM::set_xml_precision, "Set the precision (number of digits) for representing floating point numbers as XML string attributes", py::args("self", "digits"))
			.def("get_xml_precision", &PyHM::get_xml_precision, "Return the precision (number of digits) for representing floating point numbers as XML string attributes", py::args("self"))
			.def("load_xml", &PyHM::load_xml, "Load a project from a XML file", py::args("self", "filename"))
			.def("set", PyHM_set_string, "Set XML attribute or value of an element. Use set('element-path..attribute', value) to set an attribute value.", py::args("self", "path", "value"))
			.def("set", PyHM_set_double, "Set a floating point property", py::args("self", "path", "value"))
			.def("set", PyHM_set_int, "Set an integer property", py::args("self", "path", "value"))
			.def("set", PyHM_set, "Set a property to an empty value", py::args("self", "path"))
			.def("set", py::raw_function(&SetParameters, 1), "Set a property in the XML tree using a path and multiple arguments or keyword arguments, i.e. set('path', x=1, y=2, z=0) is equivalent to set('path.x', 1), set('path.y', 2), set('path.z', 0)")
			.def("get", &PyHM::get, "Get XML attribute or value of an element. Use get('element-path..attribute') to get an attribute value. If the emelent does not exists returns an empty string", py::args("self", "path"))
			.def("erase", &PyHM::erase, "Remove a path from the XML tree", py::args("self", "path"))
			.def("set_timestep_callback", &PyHM::set_timestep_callback, "Set a callback function to be called each timestep of the solver. If the callback returns True, the solver is canceled.", py::args("self", "func"))
			.def("set_variable", &PyHM::set_variable, "Set a Python variable, which can be later used in XML attributes as Python expressions", py::args("self", "name", "value"))
			.def("get_variable", &PyHM::get_variable, "Get a Python variable", py::args("self", "name"))
			.def("set_log_file", &PyHM::set_log_file, "Set filename for capturing the console output", py::args("self", "filename"))
			.def("set_py_enabled", &PyHM::set_py_enabled, "Enable/Disable Python evaluation of XML attributes as Python expressions requested by the solver", py::args("self"))
		;
	}
};


#include HM_PYTHON_HEADER_NAME
// BOOST_PYTHON_MODULE(heamd)
{
	PyHMModule module;
}


//! exception handling routine for std::set_terminate
void exception_handler()
{
	static bool tried_throw = false;

	#pragma omp critical
	{

	LOG_CERR << "exception handler called" << std::endl;

	try {
		if (!tried_throw) throw;
		LOG_CERR << "no active exception" << std::endl;
	}
	catch (boost::exception& e) {
		LOG_CERR << "boost::exception: " << boost::diagnostic_information(e) << std::endl;
	}
	catch (std::exception& e) {
		LOG_CERR << "std::exception: " << e.what() << std::endl;
	}
	catch (std::string& e) {
		LOG_CERR << e << std::endl;
	}
	catch(const char* e) {
		LOG_CERR << e << std::endl;
	}
	catch(py::error_already_set& e) {
		//PyObject *ptype, *pvalue, *ptraceback;
		//PyErr_Fetch(&ptype, &pvalue, &ptraceback);
		//char* pStrErrorMessage = PyString_AsString(pvalue);
		//LOG_CERR << "Python error: " << pStrErrorMessage << std::endl;
		LOG_CERR << "Python error" << std::endl;
	}
	catch(...) {
		LOG_CERR << "Unknown error" << std::endl;
	}

	// print date/time
	LOG_CERR << "Local timestamp: " << boost::posix_time::second_clock::local_time() << std::endl;

	print_stacktrace(LOG_CERR);

	// restore teminal colors
	LOG_COUT << DEFAULT_TEXT << std::endl;

	} // critical

	// exit app with failure code
	exit(EXIT_FAILURE);
}


//! Run test routines
template<typename T, typename P, int DIM>
int run_tests()
{
	int nfail = 0;

	LOG_COUT << "Running tests for T=" << typeid(T).name() << " P=" << typeid(P).name() << " DIM=" << DIM << "..." << std::endl;

	{
		LOG_COUT << "\n# Test 1" << std::endl;
		nfail += 0;
	}

	if (nfail == 0) {
		LOG_COUT << GREEN_TEXT << "ALL TESTS PASSED" << DEFAULT_TEXT << std::endl;
	}
	else if (nfail == 1) {
		LOG_COUT << RED_TEXT << "1 TEST FAILED" << DEFAULT_TEXT << std::endl;
	}
	else {
		LOG_COUT << RED_TEXT << nfail << " TESTS FAILED" << DEFAULT_TEXT << std::endl;
	}

	return nfail;
}


//#include "checkcpu.h"


//! main entry point of application
int main(int argc, char* argv[])
{
	// read program arguments
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("test", "run tests")
	    ("disable-python", "disable Python code evaluation in project files")
	    ("input-file", po::value< std::string >()->default_value("project.xml"), "input file")
	    ("actions-path", po::value< std::string >()->default_value("actions"), "actions xpath to run in input file")
	;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
		  options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		// print help
		LOG_COUT << desc << "\n";
		return 1;
	}

	// set exception handler
	std::set_terminate(exception_handler);

	// init python
	Py_Initialize();

	// run some small problems for checking correctness
	if (vm.count("test")) {
		return run_tests<double, double, 3>();
	}

#if 0
	// check CPU features (only informative)
	if (can_use_intel_core_4th_gen_features()) {
		LOG_COUT << GREEN_TEXT << BOLD_TEXT << "Info: This CPU supports ISA extensions introduced in Haswell!" << std::endl;
	}
#endif

	// run the app
	PyHMModule module;
	py::object pyhm = module.HM();
	PyHM& hmp = py::extract<PyHM&>(pyhm);
	hmp.set_py_enabled(vm.count("disable-python") == 0);
	int ret = hmp.exec(vm);
	hmp.reset();

	// return exit code
	return ret;
}

