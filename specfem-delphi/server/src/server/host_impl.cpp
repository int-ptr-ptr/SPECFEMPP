#ifndef _DELPHI_SERVER_HOST_IMPL_CPP
#define _DELPHI_SERVER_HOST_IMPL_CPP

#include "server/host_impl.hpp"

// beast library reference
// https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/quickref.html

specfem::delphi::host<true>::impl::impl(
    boost::asio::ip::tcp::endpoint endpoint) {}
specfem::delphi::host<false>::impl::impl(
    boost::asio::ip::tcp::endpoint endpoint)
    : io_context(), ws(io_context), socket(io_context), endpoint(endpoint),
      acceptor(io_context, boost::asio::ip::tcp::v4()) {
  acceptor.bind(endpoint);
  acceptor.listen();
}
void specfem::delphi::host<false>::impl::sync_await_connection() {
  acceptor.accept(boost::beast::get_lowest_layer(ws).socket());
  socket.connect(endpoint);
}
void specfem::delphi::host<false>::impl::sync_accept_websocket_upgrade() {
  ws.accept();
}
void specfem::delphi::host<false>::impl::sync_await_read() {
  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/ref/boost__beast__http__read/overload1.html
  ws.read(buffer, error_code);

  //==== DEMO CODE BELOW ====
  if (error_code == boost::beast::error::timeout) {
    return;
  }
  if (error_code == boost::beast::http::error::end_of_stream) {
    return;
  }
  if (error_code == boost::beast::http::error::partial_message) {
    return;
  }
  if (error_code == boost::beast::websocket::error::closed) {
    return;
  }

  // Set text mode if the received message was also text,
  // otherwise binary mode will be set.
  ws.text(ws.got_text());

  // Echo the received message back to the peer. If the received
  // message was in text mode, the echoed message will also be
  // in text mode, otherwise it will be in binary mode.
  ws.write(buffer.data(), error_code);
  if (error_code == boost::beast::error::timeout) {
    return;
  }
  if (error_code == boost::beast::http::error::end_of_stream) {
    return;
  }
  if (error_code == boost::beast::http::error::partial_message) {
    return;
  }

  // Discard all of the bytes stored in the dynamic buffer,
  // otherwise the next call to read will append to the existing
  // data instead of building a fresh message.
  buffer.consume(buffer.size());
}
bool specfem::delphi::host<false>::impl::is_open() {
  return error_code != boost::beast::error::timeout &&
         error_code != boost::beast::http::error::end_of_stream &&
         error_code != boost::beast::http::error::partial_message &&
         error_code != boost::beast::websocket::error::closed;
}

#endif
