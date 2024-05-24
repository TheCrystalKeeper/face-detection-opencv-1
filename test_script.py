import unittest

def add(a, b):
    """Function to add two numbers"""
    return a + b

class TestAddFunction(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)
    
    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)
    
    def test_add_positive_and_negative(self):
        self.assertEqual(add(2, -3), -1)

if __name__ == '__main__':
    unittest.main()
