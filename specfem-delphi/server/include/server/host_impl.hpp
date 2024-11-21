#ifndef _DELPHI_SERVER_HOST_IMPL_HPP
#define _DELPHI_SERVER_HOST_IMPL_HPP

#include "delphi.hpp"
#include "host.hpp"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
// If we implement ssl, we will also need these:
//  #include <boost/asio/ssl.hpp>
//  #include <boost/beast/websocket/ssl.hpp>

class specfem::delphi::host<true>::impl {
public:
  impl(boost::asio::ip::tcp::endpoint endpoint);
};

class specfem::delphi::host<false>::impl {
public:
  impl(boost::asio::ip::tcp::endpoint endpoint);

  bool is_open();
  void sync_await_connection();
  // TODO handle any HTTP request?
  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/using_websocket/handshaking.html
  void sync_accept_websocket_upgrade();
  void sync_await_read();

  boost::beast::error_code error_code;
  boost::asio::io_context io_context;

  // https://www.boost.org/doc/libs/1_86_0/doc/html/boost_asio/reference/ip__tcp/endpoint.html
  boost::asio::ip::tcp::endpoint endpoint;

  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/ref/boost__beast__tcp_stream.html
  boost::beast::tcp_stream socket;

  boost::asio::ip::tcp::acceptor acceptor;

  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/ref/boost__beast__websocket__stream.html
  boost::beast::websocket::stream<boost::beast::tcp_stream> ws;

  boost::beast::flat_buffer buffer;
};

#endif
