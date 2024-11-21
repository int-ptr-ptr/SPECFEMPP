#ifndef _DELPHI_SERVER_HOST_CPP
#define _DELPHI_SERVER_HOST_CPP

#include "server/host_impl.hpp"

// beast library reference
// https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/quickref.html

static boost::asio::ip::tcp::endpoint get_endpoint(std::string endpoint,
                                                   unsigned int port) {
  if (endpoint == "localhost") {
    return boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port);
  } else {
    throw std::runtime_error("Cannot parse endpoint string!");
  }
}

// specfem::delphi::host<false>::host(std::string endpoint, unsigned int port):
// pImpl(new specfem::delphi::host<false>::impl(get_endpoint(endpoint,port))){}
// specfem::delphi::host<true>::host(std::string endpoint, unsigned int port):
// pImpl(new specfem::delphi::host<true>::impl(get_endpoint(endpoint,port))){}

specfem::delphi::host<false>::host(std::string endpoint, unsigned int port)
    : pImpl(std::make_shared<specfem::delphi::host<false>::impl>(
          get_endpoint(endpoint, port))) {}
specfem::delphi::host<true>::host(std::string endpoint, unsigned int port)
    : pImpl(std::make_shared<specfem::delphi::host<true>::impl>(
          get_endpoint(endpoint, port))) {}

bool specfem::delphi::host<false>::is_open() { return pImpl->is_open(); }
void specfem::delphi::host<false>::sync_await_connection() {
  pImpl->sync_await_connection();
}
void specfem::delphi::host<false>::sync_accept_websocket_upgrade() {
  pImpl->sync_accept_websocket_upgrade();
}
void specfem::delphi::host<false>::sync_await_read() {
  pImpl->sync_await_read();
}

#endif
