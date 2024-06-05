from helper import *
import pytest

def test_sort_coulomb_matrices():
    coulomb_matrices = np.array([
        # 3 2 1
        [[1, 2, 3], 
         [2, 1, 4], 
         [3, 4, 1]], #
        [[1, 2, 4], 
         [2, 1, 3], 
         [4, 3, 1]], # permute 1,2
        [[1, 3, 4], 
         [3, 1, 2], 
         [4, 2, 1]], # permute 1,3
        [[1, 4, 3], 
         [4, 1, 2], 
         [3, 2, 1]], # permute 2,3
        [[4, 5, 6], 
         [5, 4, 7], 
         [6, 7, 4]]
    ])
    
    expected_output = np.array([
        [1, 4, 3, 1, 2, 1],
        [1, 4, 3, 1, 2, 1],
        [1, 4, 3, 1, 2, 1],
        [1, 4, 3, 1, 2, 1],
        [4, 7, 6, 4, 5, 4]
    ])
    
    output = sort_coulomb_matrices(coulomb_matrices)

    np.testing.assert_array_equal(output, expected_output)

# Command to run the test
if __name__ == "__main__":
    pytest.main()