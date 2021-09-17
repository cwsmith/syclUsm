#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>

struct LOs {
  int* buf;
  int n;
  LOs() : buf(nullptr) {}
  //non-default ctor is OK
  LOs(int len) {
    auto policy = oneapi::dpl::execution::dpcpp_default;
    buf = sycl::malloc_device<int>(len,policy);
    n=len;
  }
  //non-default copy ctor trips the trait asserts
  LOs(const LOs&) {
  }
  int& operator[](int i) const {
    return buf[i];
  }
};

#define checkType(T,name) \
static_assert(std::is_trivially_move_constructible<T>::value, name " is not trivially move constructible"); \
static_assert(std::is_trivially_copy_constructible<T>::value, name " is not trivially copy constructible"); \
static_assert(std::is_trivially_destructible<T>::value, name " is not trivially destructible"); \
static_assert(std::is_trivially_move_assignable<T>::value, name " is not trivially move assignable"); \
static_assert(std::is_trivially_copy_assignable<T>::value, name " is not trivially copy constructible"); \
static_assert(std::is_standard_layout<T>::value, name " is not in the standard layout"); \
static_assert(std::is_trivially_copyable<T>::value, name " is not trivially_copyable"); \

//uncomment the following to see which traits are not satisfied
//checkType(LOs, "LOs");
#undef checkType

//borrowed from Kokkos core/src/SYCL/Kokkos_SYCL_Instance.hpp:
template <typename T>
T& copy_from(const T& t, sycl::queue& q) {
  auto policy = oneapi::dpl::execution::dpcpp_default;
  void* m_data = sycl::malloc_shared<int>(sizeof(T),policy);
  q.memcpy(m_data, std::addressof(t), sizeof(T));
  q.wait();
  assert(m_data);
  return *reinterpret_cast<T*>(m_data);
}

void check(sycl::queue& q, LOs& a) {
  int* bufH = (int*) malloc(a.n*sizeof(int));
  try {
    q.memcpy(bufH,a.buf,a.n*sizeof(int));
    q.wait();
  } catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
  }   
  for( int i=0; i<a.n; i++) {
    assert(bufH[i] == 42);
  }
  free(bufH);
}

//TODO: fix memory leaks - see 
//  https://github.com/pvelesko/SYCL_Tutorials/blob/master/Porting%20Object%20Oriented%20Code%20to%20USM.md#destructor
int main() {
  const int n=1024;
  auto policy = oneapi::dpl::execution::dpcpp_default;
  LOs a;
  a.buf = sycl::malloc_device<int>(n,policy);
  a.n = n;
  assert(a.buf);

  auto answer = [=](int i) {
    a[i] = 42;
  };

  sycl::queue q; //uses default
  const auto functor = copy_from(answer,q);
  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class bar>( sycl::range<1>{n}, functor);
    }); 
    q.wait();
  } catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
