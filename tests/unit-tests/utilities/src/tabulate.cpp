#include "utilities/include/tabulate.hpp"

static void pad_string(std::ostream &stream, const std::string &str,
                       int length) {
  int remainder = std::max(0, length - (int)str.length());
  stream << std::string(remainder / 2, ' ') << str
         << std::string(remainder - remainder / 2, ' ');
}

std::ostream &std::operator<<(std::ostream &stream,
                              const specfem::test::Table &table) {

  std::vector<int> col_widths;
  int row_label_length = 0;

  if (table.has_row_labels()) {
    for (int irow = 0; irow < table.get_nrows(); ++irow) {
      row_label_length =
          std::max(row_label_length, (int)table.get_row_label(irow).length());
    }
  }

  for (int icol = 0; icol < table.get_ncols(); ++icol) {
    col_widths.push_back(table.width_of_column(icol));
  }

  const auto print_row_divider = [&](bool has_above, bool has_below) {
    stream << (has_above ? (has_below ? "├" : "└") : (has_below ? "┌" : "╴"));
    std::string central =
        has_above ? (has_below ? "┼" : "┴") : (has_below ? "┬" : "─");

    if (row_label_length > 0) {
      for (int i = 0; i < row_label_length; ++i) {
        stream << "─";
      }
      stream << central;
    }

    // fill each column
    int icol;
    for (icol = 0; icol < table.get_ncols() - 1; ++icol) {

      for (int i = 0; i < col_widths[icol]; ++i) {
        stream << "─";
      }
      stream << central;
    }
    for (int i = 0; i < col_widths[icol]; ++i) {
      stream << "─";
    }

    stream << (has_above ? (has_below ? "┤" : "┘") : (has_below ? "┐" : "╶"))
           << '\n';
  };

  // top
  print_row_divider(false, true);

  // labels
  if (table.has_column_labels()) {
    stream << "│";

    if (row_label_length > 0) {
      for (int i = 0; i < row_label_length; ++i) {
        stream << " ";
      }
      stream << "│";
    }

    for (int icol = 0; icol < table.get_ncols(); ++icol) {
      pad_string(stream, table.get_column_label(icol), col_widths[icol]);
      stream << "│";
    }
    stream << "\n";

    print_row_divider(true, true);
  }

  for (int irow = 0; irow < table.get_nrows(); ++irow) {
    // content
    std::vector<std::vector<std::string> > data;
    for (int icol = 0; icol < table.get_ncols(); ++icol) {
      data.push_back(table.data_as_lines(irow, icol));
    }
    int rowheight = table.height_of_row(irow);

    for (int isub = 0; isub < rowheight; ++isub) {

      stream << "│";

      if (row_label_length > 0) {
        if (isub == (rowheight / 2)) {
          pad_string(stream, table.get_row_label(irow), row_label_length);
        } else {
          pad_string(stream, std::string(), row_label_length);
        }

        stream << "│";
      }

      for (int icol = 0; icol < table.get_ncols(); ++icol) {
        pad_string(stream,
                   (data[icol].size() > isub) ? data[icol][isub]
                                              : std::string(),
                   col_widths[icol]);
        stream << "│";
      }
      stream << "\n";
    }

    // divider
    print_row_divider(true, irow < table.get_nrows() - 1);
  }
  stream << std::flush;

  return stream;
}

std::string std::to_string(const specfem::test::Table &table) {
  std::ostringstream oss;
  oss << table;
  return std::move(oss).str();
}

int specfem::test::Table::insert_column(int position) {
  if (position < 0) {
    position = ncols + 1 + position;
  }

  if (position > ncols) {
    std::ostringstream oss;
    oss << "Cannot insert column into table with " << ncols
        << " columns at position " << position << " (out of bounds)";
    throw std::runtime_error(oss.str());
  }
  if (position < 0) {
    std::ostringstream oss;
    oss << "Position " << position - ncols - 1
        << " is too negative for a table with " << ncols
        << " columns (out of bounds)";
    throw std::runtime_error(oss.str());
  }

  data.resize(nrows * (ncols + 1));

  // shift data (highest index first, so no overwriting happens)
  for (int i = nrows * ncols - 1; i >= 0; --i) {
    const auto [irow, icol] = to_2d_index(i);
    if (icol >= position) {
      ncols++;
      data[from_2d_index(irow, icol + 1)] = std::move(data[i]);
      ncols--;
      // ptr in old location is killed for us
    }
  }
  ncols++;
  column_formats.emplace(column_formats.begin() + position);
  column_labels.emplace(column_labels.begin() + position);

  return position;
}

int specfem::test::Table::insert_row(int position) {
  if (position < 0) {
    position = nrows + 1 + position;
  }

  if (position > nrows) {
    std::ostringstream oss;
    oss << "Cannot insert row into table with " << nrows << " rows at position "
        << position << " (out of bounds)";
    throw std::runtime_error(oss.str());
  }
  if (position < 0) {
    std::ostringstream oss;
    oss << "Position " << position - nrows - 1
        << " is too negative for a table with " << nrows
        << " rows (out of bounds)";
    throw std::runtime_error(oss.str());
  }

  data.resize((nrows + 1) * ncols);

  // shift data (highest index first, so no overwriting happens)
  for (int i = nrows * ncols - 1; i >= 0; --i) {
    const auto [irow, icol] = to_2d_index(i);
    if (irow >= position) {
      nrows++;
      data[from_2d_index(irow + 1, icol)] = std::move(data[i]);
      nrows--;
      // ptr in old location is killed for us
    }
  }

  nrows++;
  row_formats.emplace(row_formats.begin() + position);
  row_labels.emplace(row_labels.begin() + position);

  return position;
}

int specfem::test::Table::width_of_column(const int &column) const {
  // this can be optimized / bugfixed. for now, take the largest string length
  int len = 0;
  for (int irow = 0; irow < nrows; ++irow) {
    const auto &lines = data_as_lines(irow, column);
    for (const auto &line : lines) {
      len = std::max(len, (int)line.length());
    }
  }
  return len;
}
int specfem::test::Table::height_of_row(const int &row) const {
  // this can be optimized / bugfixed. for now, take the largest string length
  int len = 0;
  for (int icol = 0; icol < ncols; ++icol) {
    len = std::max(len, (int)data_as_lines(row, icol).size());
  }
  return len;
}
