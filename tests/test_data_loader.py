import os
import pandas as pd
import pytest
from src.data_loader import load_image_paths

def create_test_files(test_dir, sub_dirs, n_files=2, file_ext='.jpeg'):
    """
    creates a set of test files in the specified subdirectories for testing
    """
    for sub_dir, _ in sub_dirs:
        os.makedirs(os.path.join(test_dir, sub_dir), exist_ok=True)
        for i in range(n_files):
            with open(os.path.join(test_dir, sub_dir, f'test{i}{file_ext}'), 'w') as f:
                f.write('test')

@pytest.fixture
def setup_test_env(tmp_path):
    """
    pytest fixture to set up a test environment with test files
    """
    test_dir = tmp_path
    sub_dirs = [('test-neg', 0), ('test-pos', 1)]
    create_test_files(test_dir, sub_dirs)
    return str(test_dir), sub_dirs

def test_load_image_paths(setup_test_env):
    """
    Test the load_image_paths function.
    """
    test_dir, sub_dirs = setup_test_env
    df = load_image_paths(test_dir, sub_dirs)

    assert isinstance(df, pd.DataFrame), "the function should return a df"
    assert len(df) == 4, "df should contain 4 rows (2 images per sub-directory)"
    assert set(df.columns) == {'image_path', 'label'}, "df should have image_path and label columns"
    assert all(df['label'].isin([0, 1])), "labels should be either 0 or 1"

def test_empty_directory(tmp_path):
    """
    test the function with an empty directory
    """
    df = load_image_paths(str(tmp_path), [('empty', 0)])
    assert len(df) == 0, "df should be empty for an empty directory"

def test_non_existent_directory(tmp_path):
    """
    test the function with a non-existent directory
    """
    df = load_image_paths(str(tmp_path), [('non-existent', 0)])
    assert len(df) == 0, "df should be empty for a non-existent directory."
    assert list(df.columns) == ['image_path', 'label'], "df should have image_path and label columns."
