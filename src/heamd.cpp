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
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>

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
#define _CLEAR_EOL	"\033[K"

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
		: _max(max), _dp(100/steps), _p(0), _p_old(-1)
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
	bool complete()
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

	//! Returns the current progress message as stream to cout
	std::ostream& message()
	{
		T percent = _p/_max*100;
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
	int Z;	// atomic number
};



template <typename T, int DIM>
class ElementDatabase
{
	typedef std::map<std::string, boost::shared_ptr<Element<T,DIM> > > ElementMap;

	ElementMap map;

public:
	// reads element objects form xml file
	void load(const std::string& filename)
	{

	}

	boost::shared_ptr<Element<T,DIM> > get(const std::string& key)
	{
	}
};





template <typename T, int DIM>
class MoleculeBase
{
};


template <typename T, int DIM>
class Molecule : public MoleculeBase<T, DIM>
{
public:
	ublas::c_vector<T, DIM> x;	// position
	ublas::c_vector<T, DIM> x0;	// previous position
	ublas::c_vector<T, DIM> a;	// acceleration
	ublas::c_vector<T, DIM> v;	// velocity
	boost::shared_ptr<Element<T, DIM> > element;	// base element
};


template <typename T, int DIM>
class GhostMolecule : public MoleculeBase<T, DIM>
{
	ublas::c_vector<T, DIM> t;	// translation
	boost::shared_ptr<Molecule<T, DIM> > molecule;    // base molecule 
};


template <typename T, int DIM>
class NextNeigbourSearchAlgorithm
{

};

template <typename T, int DIM>
class PoissonDisk : public NextNeigbourSearchAlgorithm<T, DIM>
{

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
};


template <typename T, int DIM>
class CubicUnitCell : public UnitCell<T, DIM>
{
protected:
	// cell dimensions
	ublas::c_vector<T, DIM> _L;
	ublas::c_vector<T, DIM> _p0;

public:
	CubicUnitCell()
	{
		for (int d = 0; d < DIM; d++) {
			_L[d] = 1.0;
			_p0[d] = 0.0;
		}
	}

	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		for (int d = 0; d < DIM; d++) {
			_L[d] = pt_get<std::size_t>(pt, (boost::format("L%s") % d).str(), _L[d]);
		}


	}

	void wrap_vector(ublas::c_vector<T, DIM>& v) const
	{
		// TODO
	}

	const ublas::c_vector<T, DIM>& bb_origin() const { return _p0; }
	const ublas::c_vector<T, DIM>& bb_size() const { return _L; }

	T volume() const
	{
		T V = _L[0];
		for (int d = 1; d < DIM; d++) {
			V *= _L[d];
		}
		return V;
	}
};


template <typename T, int DIM>
class VerletCell
{
protected:
	std::list<std::size_t> _indices;	// list with molecule indices

public:
	VerletCell()
	{
	}

	const std::list<std::size_t>& indices() const {
		return _indices;
	}

	void clear()
	{
		_indices.clear();
	}

	void add(std::size_t index)
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


template <typename T, int DIM>
class MDSolver
{
protected:
	std::string _cell_type;
	std::string _result_filename;
	std::size_t _N;
	boost::shared_ptr< UnitCell<T, DIM> > _cell;
	boost::shared_ptr< VerletMap<T, DIM> > _vmap;
	std::array<std::size_t, DIM> _vdims;
	T _nn_distance_factor;	// relative number of nearst neighbours included in the Verlet map

	std::vector< boost::shared_ptr< Molecule<T, DIM> > > _molecules;


public:
	MDSolver()
	{
		_cell_type = "cubic";
		_nn_distance_factor = 1.0;
		_result_filename = "results.xml";
	}

	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		_N = pt_get<std::size_t>(pt, "n", _N);

		_cell_type = pt_get<std::string>(pt, "cell_type", _cell_type);
		_result_filename = pt_get<std::string>(pt, "result_filename", _result_filename);

	}

	void init()
	{
		LOG_COUT << "init" << std::endl;

		if (_cell_type == "cubic") {
			_cell.reset(new CubicUnitCell<T, DIM>());
		}

		// compute number of Verlet divisions for each dimension
		const ublas::c_vector<T, DIM>& bb_size = _cell->bb_size();
		T vol_per_element = _cell->volume()/_N;
		T nn_distance = std::pow(vol_per_element, 1/(T)DIM);

		for (std::size_t d = 0; d < DIM; d++) {
			_vdims[d] = std::max((int)std::ceil(bb_size[d]/nn_distance), 1) + 2;	// +2 for periodic boundary cells
		}
		_vmap.reset(new VerletMap<T, DIM>(_vdims));

		LOG_COUT << "Verlet dimensions x = " << _vdims[0] << std::endl;

		init_molecules();
	}

	void run()
	{
		LOG_COUT << "run" << std::endl;
		
		LOG_COUT << "result file: " << _result_filename << std::endl;
		std::ofstream f;
		f.open(_result_filename);
		
		write_header(f);

		for (std::size_t i = 0; i < 5; i++) {
			write_timestep(0.0, f);
		}

		write_tailing(f);
	}

	void write_header(std::ofstream & f)
	{
		f << "<results>\n";

		f << "\t<dim>" << DIM << "</dim>\n";
		
		f << "\t<cell>\n";
		f << "\t\t<type>" << _cell_type << "</type>\n";
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();
		f << "\t\t<size";
			for (std::size_t d = 0; d < DIM; d++)
				f << (boost::format(" s%d='%g'") % d % L[d]).str();
		f << " />\n";
		f << "\t\t<origin";
			for (std::size_t d = 0; d < DIM; d++)
				f << (boost::format(" p%d='%g'") % d % p0[d]).str();
		f << " />\n";
		f << "\t</cell>\n";
	
		f << "\t<molecules>\n";
			for (std::size_t i = 0; i < _molecules.size(); i++) {
				f << (boost::format("\t\t<molecule id='%d'>") % i).str();
				f << "\t\t</molecule>\n";
			}
		f << "\t</molecules>\n";
		
		f << "\t<timesteps>\n";
	}

	void write_tailing(std::ofstream & f)
	{
		f << "\t</timesteps>\n";

		f << "</results>\n";
		f.close();
	}

	void write_timestep(T t, std::ofstream & f)
	{
		f << (boost::format("\t\t<timestep t='%g'>\n") % t).str();

		f << "\t\t\t<molecules>\n";
			for (std::size_t i = 0; i < _molecules.size(); i++) {
				f << (boost::format("\t\t\t\t<molecule id='%d'") % i).str();
				for (std::size_t d = 0; d < DIM; d++) {
					f << (boost::format(" p%d='%g'") % d % _molecules[i]->x[d]).str();
					f << (boost::format(" v%d='%g'") % d % _molecules[i]->v[d]).str();
					f << (boost::format(" a%d='%g'") % d % _molecules[i]->a[d]).str();
				}
				f << " />\n";
			}
		f << "\t\t\t</molecules>\n";
		
		f << "\t\t</timestep>\n";
	}

	
	void init_molecules()
	{
		RandomNormal01<T>& rnd = RandomNormal01<T>::instance();
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();

		// create N molecules
		std::list< boost::shared_ptr< Molecule<T, DIM> > > molecules;
		for (std::size_t i = 0; i < _N; i++)
		{
			boost::shared_ptr< Molecule<T, DIM> > m;
			m.reset(new Molecule<T, DIM>());

			// create random position and velocity
			for (std::size_t j = 0; j < DIM; j++) {
				m->x[j] = p0[j] + L[j]*rnd.rnd();
				m->a[j] = (T)0;
				m->v[j] = (T)0;
			}

			// wrap position
			_cell->wrap_vector(m->x);
			m->x0 = m->x;

			// add molecule
			molecules.push_back(m);
		}

		// init verlet map
		_molecules.resize(molecules.size());
		
		auto it = molecules.begin();
		for (std::size_t i = 0; i < molecules.size(); i++) {
			_molecules[i] = *it;
			++it;
		}

		verlet_update();
	}

	void verlet_update()
	{
		_vmap->clear();

		for (std::size_t i = 0; i < _molecules.size(); i++) {
			verlet_add(i);
		}
	}

	// compute verlet map index from position
	// position must be within the bb of the cell
	void verlet_index(const ublas::c_vector<T, DIM>& p, std::array<std::size_t, DIM>& index)
	{
		// compute verlet index
		const ublas::c_vector<T, DIM>& p0 = _cell->bb_origin();
		const ublas::c_vector<T, DIM>& L = _cell->bb_size();

		for (std::size_t i = 0; i < DIM; i++) {
			index[i] = (std::size_t)(1 + (p[i] - p0[i]) / L[i] * (_vdims[i] - 2));
			index[i] = std::max((std::size_t) 1, std::min(index[i], _vdims[i] - 2));
		}
	}

	void verlet_add(std::size_t m_index)
	{
		// compute verlet index
		boost::shared_ptr< Molecule<T, DIM> > m = _molecules[m_index];
		std::array<std::size_t, DIM> index;
		verlet_index(m->x, index);

		// add molecule to verlet cell
		boost::shared_ptr<VerletCell<T, DIM> > v_cell = _vmap->get_cell(index);
		v_cell->add(m_index);
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



//! Lippmann-Schwinger solver
template<typename T, typename P, int DIM>
class LSSolver
{
public:
	typedef boost::function<bool()> LoadstepCallback;

protected:

	// callbacks
	LoadstepCallback _loadstep_callback;

};
*/



//! heamd interface (e.g. for interfacing with Python)
class HMI
{
public:
	typedef boost::function<bool()> LoadstepCallback;

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

	//! set a loadstep callback routine (run each loadstep)
	virtual void set_loadstep_callback(LoadstepCallback cb) = 0;

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
	LoadstepCallback loadstep_callback;
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

	bool loadstep_callback_wrap()
	{
		if (!loadstep_callback) {
			return false;
		}

		return loadstep_callback();
	}

	void set_loadstep_callback(LoadstepCallback cb)
	{
		loadstep_callback = cb;
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
		//RUN_TYPE_AND_DIM(double, float, 2);
		//RUN_TYPE_AND_DIM(double, double, 3);
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
	py::object _py_loadstep_callback;

public:

	PyHM() { }
	PyHM(py::tuple args, py::dict kw) { }

	~PyHM()
	{
		//LOG_COUT << "~PyHM" << std::endl;
		PY::release();
	}

	bool loadstep_callback()
	{
		if (_py_loadstep_callback) {
			py::object ret = _py_loadstep_callback();
			py::extract<bool> bool_ret(ret);
			if (bool_ret.check()) {
				return bool_ret();
			}
		}

		return false;
	}

	void set_loadstep_callback(py::object cb)
	{
		_py_loadstep_callback = cb;
		hm()->set_loadstep_callback(boost::bind(&PyHM::loadstep_callback, this));
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
			.def("set_loadstep_callback", &PyHM::set_loadstep_callback, "Set a callback function to be called each loadstep of the solver. If the callback returns True, the solver is canceled.", py::args("self", "func"))
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

