#pragma once
#include "tabulate_impl/entry.hpp"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <iostream>
namespace specfem::test {

/**
 * @brief Manages the printing of a table of data by fixing row and column width
 * of each data entry. Both rows and columns may be labeled by a string.
 *
 *
 * By default, a table will be formatted using box characters, akin to
 * ┌───────┬──────────┬──────────┬─────┐
 * │       │ column 0 │ column 1 │ ... │
 * │───────┼──────────┼──────────┼─────│
 * │ row 0 │ row0data │ row0data │ ... │
 * │───────┼──────────┼──────────┼─────│
 * │ row 0 │ row1data │ row1data │ ... │
 * └───────┴──────────┴──────────┴─────┘
 *
 * If no rows are labeled, then the column to the left of "column 0" is skipped.
 * Similarly, if no column is labeled, then the row above "row 0" is skipped.
 *
 * Data can be of any type, with a formatter specified per row / column to
 * convert that cell to a string (more specifically, a vector of strings in the
 * case of multiline entries).
 */
class Table {
public:
  using EntryType = specfem::test::tabulate_impl::EntryType;
  using EntryFormatType = specfem::test::tabulate_impl::EntryFormat;

private:
  // =======================================
  //   Table members
  // =======================================
  int ncols; // number of columns
  int nrows; // number of rows

  std::vector<std::string> column_labels;
  std::vector<std::string> row_labels;
  bool has_column_label;
  bool has_row_label;

  std::vector<std::unique_ptr<std::any> > data; // the columns in this
                                                // table with their
                                                // data

  std::vector<EntryFormatType> column_formats;
  std::vector<EntryFormatType> row_formats;

private:
  std::pair<int, int> to_2d_index(const int &index) const {
    return std::make_pair(index / ncols, index % ncols);
  }
  int from_2d_index(const int &irow, const int &icol) const {
    return irow * ncols + icol;
  }

  /**
   * @brief Handles Python-like indexing for negative values, as well as bounds
   * checking on the row.
   */
  int handle_relative_row(int row) {

    if (row < 0) {
      row = nrows + row;
    }

    if (row > nrows) {
      std::ostringstream oss;
      oss << "Cannot index row " << row << " of table with " << nrows
          << " rows (out of bounds)";
      throw std::runtime_error(oss.str());
    }
    if (row < 0) {
      std::ostringstream oss;
      oss << "Position " << row - nrows << " is too negative for a table with "
          << nrows << " rows (out of bounds)";
      throw std::runtime_error(oss.str());
    }
    return row;
  }

  /**
   * @brief Handles Python-like indexing for negative values, as well as bounds
   * checking on the column.
   */
  int handle_relative_col(int col) {

    if (col < 0) {
      col = ncols + col;
    }

    if (col > ncols) {
      std::ostringstream oss;
      oss << "Cannot index column " << col << " of table with " << ncols
          << " columns (out of bounds)";
      throw std::runtime_error(oss.str());
    }
    if (col < 0) {
      std::ostringstream oss;
      oss << "Position " << col - ncols << " is too negative for a table with "
          << ncols << " cols (out of bounds)";
      throw std::runtime_error(oss.str());
    }
    return col;
  }

public:
  Table(const int &nrows = 0, const int &ncols = 0)
      : ncols(ncols), nrows(nrows), data(ncols * nrows), column_labels(ncols),
        row_labels(nrows), column_formats(ncols), row_formats(nrows),
        has_row_label(false), has_column_label(false) {}

  /**
   * @brief Gets the width of the given column in number of characters. Padding
   * (and dividers) not included.
   *
   * @param column Index of the column
   * @return int Width of the column, in number of characters.
   */
  int width_of_column(const int &column) const;
  /**
   * @brief Gets the height of the given row in number of characters. Padding
   * (and dividers) not included.
   *
   * @param row Index of the row
   * @return int Height of the row, in number of characters.
   */
  int height_of_row(const int &row) const;

  /**
   * @brief Inserts a column at a given index. All columns originally at or
   * after `position` are shifted by one.
   *
   * @param position index to insert into (a negative value counts back from the
   * end)
   * @return int the index (positive) of the new column.
   */
  int insert_column(int position = -1);

  /**
   * @brief Inserts a row at a given index. All rows originally at or
   * after `position` are shifted by one.
   *
   * @param position index to insert into (a negative value counts back from the
   * end)
   * @return int the index (positive) of the new row.
   */
  int insert_row(int position = -1);

  /**
   * @brief Set the data at the given row and column.
   *
   * @param row row to place data in
   * @param column column to place data in
   * @param data the data to place. This is copied into a unique_ptr.
   */
  void set_data(const int &row, const int &column, const std::any &data) {
    this->data[from_2d_index(handle_relative_row(row),
                             handle_relative_col(column))] =
        std::make_unique<std::any>(data);
  }

  void set_row_format(const int &row, const EntryType &type) {
    row_formats[handle_relative_row(row)] =
        specfem::test::tabulate_impl::formatter_from_type(type);
  }
  void set_column_format(const int &column, const EntryType &type) {
    column_formats[handle_relative_col(column)] =
        specfem::test::tabulate_impl::formatter_from_type(type);
  }
  void set_row_label(int row, const std::string &label) {
    row = handle_relative_row(row);
    row_labels[row] = label;
    if (label.length() > 0) {
      has_row_label = true;
    } else {
      // see if any labels are still set
      has_row_label = false;
      for (int irow = 0; irow < nrows; irow++) {
        if (row_labels[irow].length() > 0) {
          has_row_label = true;
          break;
        }
      }
    }
  }
  void set_column_label(int column, const std::string &label) {
    column = handle_relative_col(column);
    column_labels[column] = label;
    if (label.length() > 0) {
      has_column_label = true;
    } else {
      // see if any labels are still set
      has_column_label = false;
      for (int icol = 0; icol < ncols; icol++) {
        if (column_labels[icol].length() > 0) {
          has_column_label = true;
          break;
        }
      }
    }
  }

  EntryFormatType entry_formatter_at(const int &row, const int &column) const {
    return row_formats[row] | column_formats[column];
  }

  std::vector<std::string> data_as_lines(const int &row,
                                         const int &column) const {
    const auto &data = this->data[from_2d_index(row, column)];
    if (data == nullptr) {
      return { "" };
    }

    try {
      const auto formatter = entry_formatter_at(row, column);
      return formatter.format(*data);
    } catch (std::bad_any_cast) {
      std::ostringstream oss;
      oss << "When reading data at row " << row << " and column " << column
          << ": Failed to cast data (type = " << (*data).type().name() << ").";
      throw std::runtime_error(oss.str());
    }
  }

  int get_nrows() const { return nrows; }
  int get_ncols() const { return ncols; }
  bool has_row_labels() const { return has_row_label; }
  bool has_column_labels() const { return has_column_label; }
  const std::string &get_row_label(const int &row) const {
    return row_labels[row];
  }
  const std::string &get_column_label(const int &column) const {
    return column_labels[column];
  }
};

} // namespace specfem::test

namespace std {
std::ostream &operator<<(std::ostream &stream,
                         const specfem::test::Table &table);
std::string to_string(const specfem::test::Table &table);
} // namespace std
