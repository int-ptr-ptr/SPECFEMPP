#ifndef _DELPHI_SERVER_HOST_HPP
#define _DELPHI_SERVER_HOST_HPP

#include "delphi.hpp"

#include <memory>
#include <string>

namespace specfem {
namespace delphi {

template <bool separate_thread> class host {
public:
  host(std::string endpoint, unsigned int port);
  host(unsigned int port) : host("localhost", port) {}

  bool is_open();
};

template <> class host<true> {
public:
  host(std::string endpoint, unsigned int port);
  host(unsigned int port) : host("localhost", port) {}
  bool is_open();

private:
  struct impl;
  const std::shared_ptr<impl> pImpl;
};

template <> class host<false> {
public:
  host(std::string endpoint, unsigned int port);
  host(unsigned int port) : host("localhost", port) {}
  bool is_open();

  void sync_await_connection();

  // TODO handle any HTTP request?
  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/using_websocket/handshaking.html
  void sync_accept_websocket_upgrade();

  void sync_await_read();

private:
  struct impl;
  const std::shared_ptr<impl> pImpl;
};

} // namespace delphi
} // namespace specfem

#endif
