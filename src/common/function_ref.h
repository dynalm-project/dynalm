#pragma once

// Non-owning, non-allocating reference to a callable (like C++26
// std::function_ref). The referenced callable must outlive the call.

#include <type_traits>
#include <utility>

namespace engine {

template <typename Sig>
class FunctionRef;

template <typename R, typename... Args>
class FunctionRef<R(Args...)> {
 public:
  template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, FunctionRef>>>
  FunctionRef(F&& f)  // NOLINT(implicit)
      : obj_(const_cast<void*>(static_cast<const void*>(&f))),
        call_([](void* obj, Args... args) -> R {
          return (*static_cast<std::remove_reference_t<F>*>(obj))(std::forward<Args>(args)...);
        }) {}

  R operator()(Args... args) const { return call_(obj_, std::forward<Args>(args)...); }

 private:
  void* obj_;
  R (*call_)(void*, Args...);
};

}  // namespace engine
