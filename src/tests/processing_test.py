import sys
import warnings

sys.path.append("..")

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.helpers.data_preproccesing import (
    encode_dataset,
    remove_invalid_val,
    split_datasets,
    standardize_dataset,
)


class TestEncodeDataset:
    """Tests for the encode_dataset function"""

    @pytest.fixture
    def sample_data(self):
        """Test data with categorical and numerical columns"""
        return pd.DataFrame(
            {
                "cat1": ["A", "B", "A", "C"],
                "cat2": ["X", "Y", "X", "Z"],
                "label1": ["low", "high", "medium", "low"],
                "label2": ["yes", "no", "yes", "no"],
                "num1": [1, 2, 3, 4],
                "num2": [10.5, 20.5, 30.5, 40.5],
            }
        )

    def test_one_hot_encoding_basic(self, sample_data):
        """Basic One-Hot Encoding test"""
        result = encode_dataset(sample_data, ohe=["cat1", "cat2"], le=[])

        # Verify that original columns are removed
        assert "cat1" not in result.columns
        assert "cat2" not in result.columns

        # Verify presence of new encoded columns
        expected_cols = ["cat1_A", "cat1_B", "cat1_C", "cat2_X", "cat2_Y", "cat2_Z"]
        for col in expected_cols:
            assert col in result.columns

        # Verify binary values
        assert all(result["cat1_A"].isin([0, 1]))
        assert result["cat1_A"].sum() == 2  # A appears 2 times

    def test_label_encoding_basic(self, sample_data):
        """Basic Label Encoding test"""
        result = encode_dataset(sample_data, ohe=[], le=["label1", "label2"])

        # Verify that columns are still present but with numerical values
        assert "label1" in result.columns
        assert "label2" in result.columns

        # Verify that values are numerical
        assert result["label1"].dtype.kind in "biufc"
        assert result["label2"].dtype.kind in "biufc"

        # Verify number of unique values
        assert len(result["label1"].unique()) == 3  # low, high, medium
        assert len(result["label2"].unique()) == 2  # yes, no

    def test_combined_encoding(self, sample_data):
        """Test of combined encoding OHE + Label"""
        result = encode_dataset(sample_data, ohe=["cat1"], le=["label1"])

        # Verify OHE
        assert "cat1" not in result.columns
        assert "cat1_A" in result.columns

        # Verify Label Encoding
        assert "label1" in result.columns
        assert result["label1"].dtype.kind in "biufc"

        # Verify that other columns are intact
        assert "cat2" in result.columns
        assert "num1" in result.columns

    def test_empty_lists(self, sample_data):
        """Test with empty lists"""
        result = encode_dataset(sample_data, ohe=[], le=[])
        pd.testing.assert_frame_equal(result, sample_data)

    def test_single_category(self):
        """Test with a single category"""
        data = pd.DataFrame({"cat": ["A", "A", "A"], "num": [1, 2, 3]})
        result = encode_dataset(data, ohe=["cat"], le=[])

        assert "cat_A" in result.columns
        assert all(result["cat_A"] == 1)


class TestStandardizeDataset:
    """Tests for the standardize_dataset function"""

    @pytest.fixture
    def numeric_data(self):
        """Numerical data for tests"""
        return pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [10, 20, 30, 40, 50],
                "cat": ["A", "B", "C", "D", "E"],
                "num3": [100, 200, 300, 400, 500],
            }
        )

    def test_standardize_all_numeric(self, numeric_data):
        """Test standardization of all numerical columns"""
        result = standardize_dataset(numeric_data)

        # Verify that numerical columns have a mean close to 0 and standard deviation close to 1
        for col in ["num1", "num2", "num3"]:
            assert abs(result[col].mean()) < 1e-10  # Practically 0
            assert abs(result[col].std() - 1) < 1e-10  # Practically 1

        # Verify that categorical column is not modified
        pd.testing.assert_series_equal(result["cat"], numeric_data["cat"])

    def test_standardize_specific_columns(self, numeric_data):
        """Test standardization of specific columns"""
        result = standardize_dataset(numeric_data, col=["num1", "num2"])

        # Verify standardization of specified columns
        assert abs(result["num1"].mean()) < 1e-10
        assert abs(result["num2"].mean()) < 1e-10

        # Verify that num3 is not standardized
        pd.testing.assert_series_equal(result["num3"], numeric_data["num3"])

    def test_empty_dataset(self):
        """Test with an empty dataset"""
        empty_df = pd.DataFrame()
        result = standardize_dataset(empty_df)
        assert result.empty

    def test_single_value_column(self):
        """Test with a column having a single unique value"""
        data = pd.DataFrame({"const": [5, 5, 5, 5]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warnings
            result = standardize_dataset(data)

        # The constant column should become 0 everywhere
        assert all(result["const"] == 0)


class TestRemoveInvalidVal:
    """Tests for the remove_invalid_val function"""

    @pytest.fixture
    def data_with_invalid(self):
        """Data with invalid values"""
        return pd.DataFrame(
            {
                "num_with_inf": [1, 2, np.inf, 4, -np.inf],
                "num_with_nan": [1, 2, np.nan, 4, 5],
                "num_clean": [1, 2, 3, 4, 5],
                "cat_with_nan": ["A", "B", np.nan, "D", "E"],
            }
        )

    def test_handle_infinite_values(self, data_with_invalid):
        """Test handling of infinite values"""
        result = remove_invalid_val(data_with_invalid, k=10)

        # Verify that there are no more infinite values
        assert not np.isinf(result["num_with_inf"]).any()

        # Verify replaced values
        max_val = abs(np.array([1, 2, 4])).max()  # max of non-infinite values
        assert result["num_with_inf"].iloc[2] == 10 * max_val  # inf replaced
        assert result["num_with_inf"].iloc[4] == -10 * max_val  # -inf replaced

    def test_handle_nan_numeric(self, data_with_invalid):
        """Test handling of NaN in numerical columns"""
        result = remove_invalid_val(data_with_invalid)

        # Verify that there are no more NaN
        assert not result["num_with_nan"].isna().any()

        # Verify that NaN is replaced by the mean
        expected_mean = np.array([1, 2, 4, 5]).mean()
        assert result["num_with_nan"].iloc[2] == expected_mean

    def test_handle_nan_categorical(self, data_with_invalid):
        """Test handling of NaN in categorical columns"""
        result = remove_invalid_val(data_with_invalid)

        # Verify that there are no more NaN
        assert not result["cat_with_nan"].isna().any()

        # Verify that NaN is replaced by the mode
        # The mode should be one of the existing values
        assert result["cat_with_nan"].iloc[2] in ["A", "B", "D", "E"]

    def test_clean_data_unchanged(self, data_with_invalid):
        """Test that clean data is not modified"""
        result = remove_invalid_val(data_with_invalid)
        pd.testing.assert_series_equal(
            result["num_clean"], data_with_invalid["num_clean"]
        )

    def test_different_k_values(self):
        """Test with different k values"""
        data = pd.DataFrame({"col": [1, np.inf, -np.inf, 2]})

        result_k5 = remove_invalid_val(data, k=5)
        result_k20 = remove_invalid_val(data, k=20)

        # Replacement values must be different
        assert abs(result_k5["col"].iloc[1]) < abs(result_k20["col"].iloc[1])


class TestSplitDatasets:
    """Tests for the split_datasets function"""

    @pytest.fixture
    def sample_dataset(self):
        """Dataset for split tests"""
        return pd.DataFrame({"A": range(10), "B": range(10, 20), "C": ["x"] * 10})

    def test_basic_split(self, sample_dataset):
        """Basic split test"""
        indexes = [3, 7]  # Split at index 3 and 7
        result = split_datasets(sample_dataset, indexes)

        # Verify number of datasets
        assert len(result) == 3

        # Verify sizes
        assert len(result[0]) == 4  # indices 0-3
        assert len(result[1]) == 4  # indices 4-7
        assert len(result[2]) == 2  # indices 8-9

        # Verify values
        assert result[0]["A"].tolist() == [0, 1, 2, 3]
        assert result[1]["A"].tolist() == [4, 5, 6, 7]
        assert result[2]["A"].tolist() == [8, 9]

    def test_single_split(self, sample_dataset):
        """Test with a single split point"""
        indexes = [4]
        result = split_datasets(sample_dataset, indexes)

        assert len(result) == 2
        assert len(result[0]) == 5
        assert len(result[1]) == 5

    def test_no_split(self, sample_dataset):
        """Test without split (returns original dataset)"""
        indexes = []
        result = split_datasets(sample_dataset, indexes)

        assert len(result) == 1
        pd.testing.assert_frame_equal(result[0], sample_dataset)

    def test_edge_cases(self, sample_dataset):
        """Test edge cases"""
        # Split at the beginning
        indexes = [0]
        result = split_datasets(sample_dataset, indexes)
        assert len(result[0]) == 1
        assert len(result[1]) == 9

        # Split at the end
        indexes = [8]
        result = split_datasets(sample_dataset, indexes)
        assert len(result[0]) == 9
        assert len(result[1]) == 1


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_complete_preprocessing_pipeline(self):
        """Test of complete preprocessing pipeline"""
        # Data with all types of problems
        data = pd.DataFrame(
            {
                "cat": ["A", "B", "A", "C"],
                "label": ["low", "high", "low", "medium"],
                "num1": [1, np.inf, 3, np.nan],
                "num2": [10, 20, 30, 40],
            }
        )

        # Complete pipeline
        # 1. Clean invalid values
        clean_data = remove_invalid_val(data)

        # 2. Encode categorical variables
        encoded_data = encode_dataset(clean_data, ohe=["cat"], le=["label"])

        # 3. Standardize
        standardized_data = standardize_dataset(encoded_data)

        # 4. Split
        train, test = split_datasets(standardized_data, [2])

        # Verifications
        assert not np.isinf(
            standardized_data.select_dtypes(include=[np.number]).values
        ).any()
        assert not np.isnan(
            standardized_data.select_dtypes(include=[np.number]).values
        ).any()
        assert len(train) == 3
        assert len(test) == 1


# Pytest configuration
@pytest.fixture(scope="session")
def setup_warnings():
    """Warning configuration for tests"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
