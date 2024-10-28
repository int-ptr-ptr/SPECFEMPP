#ifndef _SPECFEM_EVENT_MARCHING_TIMESCHEME_WRAPPER_HPP
#define _SPECFEM_EVENT_MARCHING_TIMESCHEME_WRAPPER_HPP

#include "event_marcher.hpp"
#include "event.hpp"

#include "timescheme/timescheme.hpp"

#include <unordered_map>
#include <set>

namespace specfem {
namespace event_marching {

template <typename TimeScheme>
class timescheme_wrapper;


template <typename TimeScheme>
class timescheme_wrapper_event: public specfem::event_marching::event{
public:
  timescheme_wrapper<TimeScheme>& wrapper;
protected:
  timescheme_wrapper_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::event_marching::precedence p):
      wrapper(wrapper),
      specfem::event_marching::event(p){}
};

template <typename TimeScheme>
class timescheme_step_event: public timescheme_wrapper_event<TimeScheme>{
public:
  timescheme_step_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::event_marching::precedence p):
      specfem::event_marching::timescheme_wrapper_event<TimeScheme>(wrapper,p){}

  int call() {
    return timescheme_wrapper_event<TimeScheme>::wrapper.step();
  }
};

template <typename TimeScheme, specfem::element::medium_tag medium, specfem::wavefield::type WaveFieldType,
        specfem::dimension::type DimensionType, typename qp_type>
class wavefield_update_event: public timescheme_wrapper_event<TimeScheme>{
public:
  wavefield_update_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels,
        specfem::event_marching::precedence p):
      kernels(kernels),
      specfem::event_marching::timescheme_wrapper_event<TimeScheme>(wrapper,p){}

  int call() {
    kernels.template update_wavefields<medium>(
        timescheme_wrapper_event<TimeScheme>::wrapper.get_istep());
    return 0;
  }

private:
  specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels;
};

template <typename TimeScheme>
class forward_predictor_event: public timescheme_wrapper_event<TimeScheme>{
public:
  forward_predictor_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::element::medium_tag medium,
        specfem::event_marching::precedence p):
      medium(medium),
      specfem::event_marching::timescheme_wrapper_event<TimeScheme>(wrapper,p){}

  int call() {
    timescheme_wrapper_event<TimeScheme>::wrapper.timescheme
        .apply_predictor_phase_forward(medium);
    return 0;
  }

private:
  specfem::element::medium_tag medium;
};
template <typename TimeScheme>
class forward_corrector_event: public timescheme_wrapper_event<TimeScheme>{
public:
  forward_corrector_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::element::medium_tag medium,
        specfem::event_marching::precedence p):
      medium(medium),
      specfem::event_marching::timescheme_wrapper_event<TimeScheme>(wrapper,p){}

  int call() {
    timescheme_wrapper_event<TimeScheme>::wrapper.timescheme
        .apply_corrector_phase_forward(medium);
    return 0;
  }

private:
  specfem::element::medium_tag medium;
};


template <typename TimeScheme, specfem::wavefield::type WaveFieldType,
        specfem::dimension::type DimensionType, typename qp_type>
class seismogram_update_event: public timescheme_wrapper_event<TimeScheme>{
public:
  seismogram_update_event(timescheme_wrapper<TimeScheme>& wrapper,
        specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels,
        specfem::event_marching::precedence p):
      kernels(kernels),
      specfem::event_marching::timescheme_wrapper_event<TimeScheme>(wrapper,p){}

  int call() {
    if (timescheme_wrapper_event<TimeScheme>::wrapper.timescheme.compute_seismogram(timescheme_wrapper_event<TimeScheme>::wrapper.get_istep())) {
      kernels.compute_seismograms(timescheme_wrapper_event<TimeScheme>::wrapper.timescheme.get_seismogram_step());
      timescheme_wrapper_event<TimeScheme>::wrapper.timescheme.increment_seismogram_step();
    }
    return 0;
  }

private:
  specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels;
};


template <typename TimeScheme>
class timescheme_wrapper{
public:

  timescheme_wrapper(TimeScheme & timescheme): timescheme(timescheme),
      istep(0), step_event(*this,specfem::event_marching::DEFAULT_EVENT_PRECEDENCE) {}
  void set_forward_predictor_event(specfem::element::medium_tag medium, precedence p);
  void set_forward_corrector_event(specfem::element::medium_tag medium, precedence p);
  
  template <specfem::element::medium_tag medium, specfem::wavefield::type WaveFieldType,
          specfem::dimension::type DimensionType, typename qp_type>
  void set_wavefield_update_event(
      specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels, precedence p);

  template <specfem::wavefield::type WaveFieldType,
          specfem::dimension::type DimensionType, typename qp_type>
  void set_seismogram_update_event(
        specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels, precedence p);

  void register_under_marcher(specfem::event_marching::event_marcher* marcher);
  void unregister_from_marcher(specfem::event_marching::event_marcher* marcher);

  int step(){
    std::cout << "step "<<istep << "\n";
    int retval = time_stepper.march_events();
    istep++;
    return retval;
  }

  TimeScheme & timescheme;
  int get_istep(){
    return istep;
  }
private:
  specfem::event_marching::event_marcher time_stepper;
  int istep;

  std::unordered_map<specfem::element::medium_tag, std::unique_ptr<specfem::event_marching::forward_predictor_event<TimeScheme>>>
      forward_predictor_events;
  std::unordered_map<specfem::element::medium_tag, std::unique_ptr<specfem::event_marching::forward_corrector_event<TimeScheme>>>
      forward_corrector_events;
  std::unordered_map<specfem::element::medium_tag, std::unique_ptr<specfem::event_marching::timescheme_wrapper_event<TimeScheme>>>
      wavefield_update_events;
  std::unordered_map<specfem::wavefield::type, std::unique_ptr<specfem::event_marching::timescheme_wrapper_event<TimeScheme>>>
      seismogram_update_events;

  //this event gets called by this stepper's parent
  specfem::event_marching::timescheme_step_event<TimeScheme> step_event;
  std::set<specfem::event_marching::event_marcher*> registrations;
};


} // namespace event_marching
} // namespace specfem

#include "timescheme_wrapper.tpp"
#endif
