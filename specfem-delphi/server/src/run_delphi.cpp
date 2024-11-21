#include <iostream>

#include "delphi.hpp"
#include "server/host.hpp"

#define DELPHI_PORT 8765

int main(int argc, char **argv) {
  specfem::delphi::host<false> host(DELPHI_PORT);
  std::cout << "Waiting for new connection.\n";
  host.sync_await_connection();
  std::cout << "Found new connection. Accepting the WS upgrade.\n";
  host.sync_accept_websocket_upgrade();
  while (host.is_open()) {
    std::cout << "waiting for message... " << std::flush;
    host.sync_await_read();
    std::cout << "received.\n";
  }
  std::cout << "connection closed\n";

  // TODO handle any HTTP request?
  // https://www.boost.org/doc/libs/1_86_0/libs/beast/doc/html/beast/using_websocket/handshaking.html
  return 0;
}
