import numpy as np
import unittest
import sys
from time import time
sys.path.insert(0, '/opt/venv/lib/python3.11/site-packages')
from dciknn import DCI


def gen_data(ambient_dim, intrinsic_dim, num_points):
    # This line generates a 2D array of random numbers.
    # intrinsic_dim -> ambient_dim 
    latent_data = 2 * np.random.rand(num_points, intrinsic_dim) - 1     # Uniformly distributed on [-1,1)
    transformation = 2 * np.random.rand(intrinsic_dim, ambient_dim) - 1
    data = np.dot(latent_data, transformation)
    return data     # num_points x ambient_dim

class TestDCI(unittest.TestCase):
    def setUp(self):
        self.dim = 5000
        self.intrinsic_dim = 50
        self.num_points = 10000
        self.num_queries = 5
        self.num_comp_indices = 2
        self.num_simp_indices = 7
        self.num_levels = 2
        self.construction_field_of_view = 10
        self.construction_prop_to_retrieve = 0.002
        self.query_field_of_view = 100
        self.query_prop_to_retrieve = 0.8
        self.data_and_query = gen_data(self.dim, self.intrinsic_dim, self.num_points + self.num_queries)
        self.data = np.copy(self.data_and_query[:self.num_points,:])
        self.query = self.data_and_query[self.num_points:,:]
        self.dci_db = DCI(self.dim, self.num_comp_indices, self.num_simp_indices)
        self.num_neighbours = 10      # The k in k-NN


    def test_add_contiguous(self):
        """
        This test case is designed to verify the construction of a DCI object and the addition of data to it.
        Raises:
            AssertionError: If the number of data points does not increase after adding the data.
        """
        print("Building DCI object... ")
        
        # Store the initial number of data points
        initial_num_points = self.dci_db.num_points
        print("initial_num_points: ", initial_num_points)
        
        print("Adding data... ")
        self.dci_db.add(self.data, num_levels=self.num_levels, field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        # Get the number of data points after adding
        final_num_points = self.dci_db.num_points
        print("data points after adding: ", final_num_points)
        
        # Assert that the number of data points has increased
        assert final_num_points > initial_num_points, "The number of data points did not increase after adding"
        print("Querying... ")


    def test_add_non_contiguous(self):
        """
        This test case is designed to verify the construction of a DCI object and the addition of sliced numpy data to it.

        The test first generates data and splits it into two parts: data and query. 
        Then, it slices the data and query numpy arrays.
        It constructs a DCI object and stores the initial number of data points.

        After that, it adds the sliced data to the DCI object and retrieves the number of data points after the addition.

        Finally, it asserts that the number of data points has increased after the addition of the sliced data.

        Raises:
            AssertionError: If the number of data points does not increase after adding the sliced data.
        """
        print("Building DCI object... ")
        
        # Slice the data and query numpy arrays
        sliced_data = self.data[:self.num_points//2, :]
        sliced_query = self.query[:self.num_queries//2, :]
        
        # Store the initial number of data points
        initial_num_points = self.dci_db.num_points
        print("initial_num_points: ", initial_num_points)
        
        print("Adding sliced data... ")
        self.dci_db.add(sliced_data, num_levels=self.num_levels, field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)
        
        # Get the number of data points after adding
        final_num_points = self.dci_db.num_points
        print("data points after adding: ", final_num_points)
        
        # Assert that the number of data points has increased
        assert final_num_points > initial_num_points, "The number of data points did not increase after adding"
        print("Querying sliced data... ")


    def test_query_contiguous(self):
        """
        """
        print("Building DCI object... ")
        # Store the initial number of data points
        self.dci_db.add(self.data, num_levels=self.num_levels, field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        print("Querying... ")
        nearest_neighbour_idx, nearest_neighbour_dists = self.dci_db.query(self.query, num_neighbours=self.num_neighbours, field_of_view=self.query_field_of_view, prop_to_retrieve=self.query_prop_to_retrieve)

        print("type of nearest_neighbour_dists", type(nearest_neighbour_idx))
        print("type of nearest_neighbour_dist", type(nearest_neighbour_dists))


    def test_query_non_contiguous(self):
        """
        This test case is designed to verify the construction of a DCI object and the querying of sliced numpy data from it.

        The test first generates data and splits it into two parts: data and query. 
        Then, it slices the data and query numpy arrays and creates a copy of the sliced data.
        It constructs a DCI object and adds the copied sliced data to it.

        After that, it queries the sliced data from the DCI object and retrieves the nearest neighbour indices and distances.

        Finally, it prints the types of the retrieved indices and distances.

        """
        print("Building DCI object... ")
        
        # Slice the data and query numpy arrays and create a copy of the sliced data
        sliced_data = np.copy(self.data[:self.num_points//2, :])
        sliced_query = np.copy(self.query[:self.num_queries//2, :])
        
        self.dci_db.add(sliced_data, num_levels=self.num_levels, field_of_view=self.construction_field_of_view, prop_to_retrieve=self.construction_prop_to_retrieve)

        print("Querying sliced data... ")
        nearest_neighbour_idx, nearest_neighbour_dists = self.dci_db.query(sliced_query, num_neighbours=self.num_neighbours, field_of_view=self.query_field_of_view, prop_to_retrieve=self.query_prop_to_retrieve)

        print("type of nearest_neighbour_dists", type(nearest_neighbour_idx))
        print("type of nearest_neighbour_dist", type(nearest_neighbour_dists))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestDCI('test_add_contiguous'))
    suite.addTest(TestDCI('test_add_non_contiguous'))
    suite.addTest(TestDCI('test_query_contiguous'))
    suite.addTest(TestDCI('test_query_non_contiguous'))
    runner = unittest.TextTestRunner()
    runner.run(suite)