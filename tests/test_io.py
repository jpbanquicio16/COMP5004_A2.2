"""Unit tests for numcompute.io."""

import os
import tempfile
import unittest

import numpy as np

from numcompute.io import iter_csv_chunks, load_csv


class _FileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write(self, text, name="data.csv"):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class TestLoadCsv(_FileTestCase):
    def test_basic_comma(self):
        p = self.write("1,2,3\n4,5,6\n")
        out = load_csv(p)
        np.testing.assert_array_equal(out, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(out.dtype, np.float64)

    def test_tab_and_semicolon_delimiters(self):
        np.testing.assert_array_equal(
            load_csv(self.write("1\t2\n3\t4\n"), delimiter="\t"), [[1, 2], [3, 4]]
        )
        np.testing.assert_array_equal(
            load_csv(self.write("1;2\n3;4\n"), delimiter=";"), [[1, 2], [3, 4]]
        )

    def test_skip_header_and_usecols(self):
        p = self.write("a,b,c\n1,2,3\n4,5,6\n")
        out = load_csv(p, skip_header=1, usecols=(0, 2))
        np.testing.assert_array_equal(out, [[1, 3], [4, 6]])

    def test_output_is_always_2d(self):
        self.assertEqual(load_csv(self.write("1,2,3\n")).shape, (1, 3))
        self.assertEqual(load_csv(self.write("1\n2\n3\n")).shape, (3, 1))

    def test_missing_keep_gives_nan(self):
        out = load_csv(self.write("1,,3\n4,5,6\n"))
        self.assertTrue(np.isnan(out[0, 1]))
        self.assertEqual(np.isnan(out).sum(), 1)

    def test_missing_tokens(self):
        out = load_csv(self.write("1,NA,3\n?,5,6\n"), missing_values=["NA", "?"])
        self.assertEqual(np.isnan(out).sum(), 2)
        self.assertTrue(np.isnan(out[0, 1]) and np.isnan(out[1, 0]))

    def test_single_string_missing_token(self):
        out = load_csv(self.write("1,NA\n3,4\n"), missing_values="NA")
        self.assertTrue(np.isnan(out[0, 1]))

    def test_omit_drops_rows_with_missing(self):
        out = load_csv(self.write("1,2\n3,\n5,6\n"), nan_policy="omit")
        np.testing.assert_array_equal(out, [[1, 2], [5, 6]])

    def test_fill_uses_placeholder(self):
        out = load_csv(self.write("1,,3\n4,5,\n"), nan_policy="fill", fill_value=-1)
        np.testing.assert_array_equal(out, [[1, -1, 3], [4, 5, -1]])

    def test_fill_requires_fill_value(self):
        with self.assertRaises(ValueError):
            load_csv(self.write("1,2\n"), nan_policy="fill")

    def test_unparseable_float_field_is_missing(self):
        out = load_csv(self.write("1,abc\n3,4\n"))
        self.assertTrue(np.isnan(out[0, 1]))

    def test_string_dtype_keeps_and_detects_missing(self):
        p = self.write("red,S\n,M\nblue,\n")
        out = load_csv(p, dtype=str)
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out[1, 0], "")
        np.testing.assert_array_equal(
            load_csv(p, dtype=str, nan_policy="omit"), [["red", "S"]]
        )

    def test_string_fill_is_not_truncated(self):
        out = load_csv(self.write("a,b\n,c\n"), dtype=str,
                       nan_policy="fill", fill_value="unknown")
        self.assertEqual(out[1, 0], "unknown")

    def test_invalid_nan_policy(self):
        with self.assertRaises(ValueError):
            load_csv(self.write("1\n"), nan_policy="drop")

    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):
            load_csv(self.write("1\n"), dtype=int)

    def test_bad_delimiter(self):
        p = self.write("1,2\n")
        with self.assertRaises(ValueError):
            load_csv(p, delimiter="")
        with self.assertRaises(TypeError):
            load_csv(p, delimiter=1)

    def test_bad_skip_header(self):
        p = self.write("1,2\n")
        with self.assertRaises(ValueError):
            load_csv(p, skip_header=-1)
        with self.assertRaises(TypeError):
            load_csv(p, skip_header="1")

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            load_csv(self.write(""))

    def test_header_only_file(self):
        with self.assertRaises(ValueError):
            load_csv(self.write("a,b\n"), skip_header=1)

    def test_ragged_rows(self):
        with self.assertRaises(ValueError):
            load_csv(self.write("1,2,3\n4,5\n"))

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_csv(os.path.join(self._tmp.name, "nope.csv"))


class TestIterCsvChunks(_FileTestCase):
    def _big(self, n=10):
        rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(n)) + "\n"
        return self.write("a,b,c\n" + rows)

    def test_chunks_have_requested_size(self):
        chunks = list(iter_csv_chunks(self._big(10), 4, skip_header=1))
        self.assertEqual([c.shape for c in chunks], [(4, 3), (4, 3), (2, 3)])

    def test_stacked_chunks_equal_load_csv(self):
        p = self._big(23)
        whole = load_csv(p, skip_header=1)
        for size in (1, 2, 5, 23, 100):
            stacked = np.vstack(list(iter_csv_chunks(p, size, skip_header=1)))
            np.testing.assert_array_equal(stacked, whole)

    def test_is_lazy_one_block_at_a_time(self):
        it = iter_csv_chunks(self._big(10), 3, skip_header=1)
        first = next(it)
        np.testing.assert_array_equal(first[0], [0, 0, 0])
        self.assertEqual(first.shape, (3, 3))

    def test_single_row_chunks_are_2d(self):
        chunks = list(iter_csv_chunks(self.write("1,2\n3,4\n"), 1))
        self.assertTrue(all(c.shape == (1, 2) for c in chunks))

    def test_missing_values_keep_and_omit(self):
        p = self.write("1,2\n3,\n5,6\n7,8\n")
        kept = np.vstack(list(iter_csv_chunks(p, 2)))
        self.assertEqual(np.isnan(kept).sum(), 1)
        omitted = np.vstack(list(iter_csv_chunks(p, 2, nan_policy="omit")))
        np.testing.assert_array_equal(omitted, [[1, 2], [5, 6], [7, 8]])

    def test_chunk_that_becomes_empty_is_skipped(self):
        p = self.write("1,2\n,\n,4\n5,6\n")
        chunks = list(iter_csv_chunks(p, 2, nan_policy="omit"))
        self.assertEqual(len(chunks), 2)
        np.testing.assert_array_equal(chunks[0], [[1, 2]])
        np.testing.assert_array_equal(chunks[1], [[5, 6]])

    def test_fill_and_string_dtype(self):
        p = self.write("a,b\n,c\nd,\n")
        out = np.vstack(list(iter_csv_chunks(
            p, 1, dtype=str, nan_policy="fill", fill_value="NA")))
        np.testing.assert_array_equal(out, [["a", "b"], ["NA", "c"], ["d", "NA"]])

    def test_header_only_file_gives_no_chunks(self):
        self.assertEqual(list(iter_csv_chunks(self.write("a,b\n"), 5, skip_header=1)), [])

    def test_bad_arguments_fail_immediately(self):
        p = self.write("1,2\n")
        with self.assertRaises(ValueError):
            iter_csv_chunks(p, 0)
        with self.assertRaises(TypeError):
            iter_csv_chunks(p, 2.5)
        with self.assertRaises(ValueError):
            iter_csv_chunks(p, 2, nan_policy="drop")
        with self.assertRaises(FileNotFoundError):
            iter_csv_chunks(os.path.join(self._tmp.name, "nope.csv"), 2)

    def test_ragged_rows_raise(self):
        with self.assertRaises(ValueError):
            list(iter_csv_chunks(self.write("1,2,3\n4,5\n"), 5))


class TestRemainingBranches(_FileTestCase):
    """Small error paths and corner cases that were not done yet."""

    def test_invalid_dtype_name_raises_typeerror(self):
        with self.assertRaises(TypeError):
            load_csv(self.write("1\n"), dtype="not-a-real-type")

    def test_missing_values_of_wrong_type(self):
        with self.assertRaises(TypeError):
            load_csv(self.write("1\n"), missing_values=5)

    def test_chunk_of_only_blank_lines_is_skipped(self):
        p = self.write("1,2\n\n\n\n3,4\n")
        chunks = list(iter_csv_chunks(p, 2))
        np.testing.assert_array_equal(np.vstack(chunks), [[1, 2], [3, 4]])


class TestUsecolsAndFillChecks(_FileTestCase):
    """The extra input checks for usecols and for fill_value."""

    def test_usecols_single_int(self):
        out = load_csv(self.write("1,2,3\n4,5,6\n"), usecols=1)
        np.testing.assert_array_equal(out, [[2], [5]])

    def test_usecols_list_and_tuple(self):
        p = self.write("1,2,3\n4,5,6\n")
        np.testing.assert_array_equal(load_csv(p, usecols=[0, 2]), [[1, 3], [4, 6]])
        np.testing.assert_array_equal(load_csv(p, usecols=(0, 2)), [[1, 3], [4, 6]])

    def test_negative_usecols_rejected(self):
        p = self.write("1,2,3\n")
        with self.assertRaises(ValueError):
            load_csv(p, usecols=-1)
        with self.assertRaises(ValueError):
            load_csv(p, usecols=[0, -1])

    def test_usecols_wrong_type(self):
        p = self.write("1,2,3\n")
        with self.assertRaises(TypeError):
            load_csv(p, usecols=1.5)
        with self.assertRaises(TypeError):
            load_csv(p, usecols=[0, 1.5])
        with self.assertRaises(TypeError):
            load_csv(p, usecols=[True])

    def test_chunk_reader_checks_usecols_immediately(self):
        with self.assertRaises(ValueError):
            iter_csv_chunks(self.write("1,2\n"), 2, usecols=-1)

    def test_fill_value_must_be_a_number_for_float_data(self):
        p = self.write("1,\n3,4\n")
        with self.assertRaises(ValueError):
            load_csv(p, nan_policy="fill", fill_value="abc")
        out = load_csv(p, nan_policy="fill", fill_value="7")
        self.assertEqual(out[0, 1], 7.0)


if __name__ == "__main__":
    unittest.main()
