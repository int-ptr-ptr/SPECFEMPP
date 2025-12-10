#include <memory>
#include <vector>
namespace specfem::test::analytical {

template <typename T> struct RoundRobinIterator {
private:
  const std::vector<std::shared_ptr<T> > *elements;
  int current_index;

public:
  RoundRobinIterator(const std::vector<std::shared_ptr<T> > *elements)
      : elements(elements), current_index(0) {}

  RoundRobinIterator() : elements(nullptr) {}

  const T &operator*() const { return *((*elements)[current_index]); }
  void operator++() { current_index = (current_index + 1) % elements->size(); }
  bool operator!=(const RoundRobinIterator<T> &other) const {
    return current_index != other.current_index;
  }
};

} // namespace specfem::test::analytical
