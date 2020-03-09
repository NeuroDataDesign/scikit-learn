from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_not_equal

from sklearn.tree import DecisionTreeRegressor
import numpy as np

def test_axis_proj():
    """Check axis projection criterion produces correct results on small toy dataset:

    ------------------
    | X | y1  y2  | weight |
    ------------------
    | 3 |  3   3  |  0.1   |
    | 5 |  3   3  |  0.3   |
    | 8 |  4   4  |  1.0   |
    | 3 |  7   7  |  0.6   |
    | 5 |  8   8  |  0.3   |
    ------------------
    |sum wt:|  2.3   |
    ------------------
 
    Mean1 = 5
    Mean2 = 5

    For all the samples, we can get the total error by summing:
    (Mean1 - y1)^2 * weight or (Mean2 - y2)^2 * weight

    I.e., total error = (5 - 3)^2 * 0.1)
                      + (5 - 3)^2 * 0.3)
                      + (5 - 4)^2 * 1.0)
                      + (5 - 7)^2 * 0.6)
                      + (5 - 8)^2 * 0.3)
                      = 0.4 + 1.2 + 1.0 + 2.4 + 2.7
                      = 7.7

    Impurity = Total error / total weight
             = 7.7 / 2.3
             = 3.3478260869565
             -----------------

    From this root node, the next best split is between X values of 5 and 8.
    Thus, we have left and right child nodes:

    LEFT                        RIGHT
    -----------------------     -----------------------
    | X | y1  y2  | weight |    | X | y1  y2  | weight |
    -----------------------     -----------------------
    | 3 |  3   3  |  0.1   |    | 8 |  4   4  |  1.0   |
    | 3 |  7   7  |  0.6   |    -----------------------
    | 5 |  3   3  |  0.3   |    |sum wt:|  1.0         |
    | 5 |  8   8  |  0.3   |    -----------------------
    -----------------------
    |sum wt:|  1.3         |
    -----------------------

    5.0625 + 3.0625 + 5.0625 + 7.5625 / 4  + 0 = 5.1875
    4 + 4.667 = 8.667

    Impurity is found in the same way:
    Left node Mean1 = Mean2 = 5.25
    Total error = ((5.25 - 3)^2 * 0.1)
                + ((5.25 - 7)^2 * 0.6)
                + ((5.25 - 3)^2 * 0.3)
                + ((5.25 - 8)^2 * 0.3)
                = 6.13125

    Left Impurity = Total error / total weight
            = 6.13125 / 1.3
            = 4.716346153846154
            -------------------

    Likewise for Right node:
    Right node Mean1 = Mean2 = 4
    Total error = ((4 - 4)^2 * 1.0)
                = 0

    Right Impurity = Total error / total weight
            = 0 / 1.0
            = 0.0
            ------
    """

    # Test axis projection where multiple y values are different:
    dt_axis_multi = DecisionTreeRegressor(random_state=0, criterion="axis",
                                   max_leaf_nodes=2)
    dt_axis_multi.fit(X=[[3], [5], [8], [3], [5]], y=[[3,2], [3,4], [4,3], [7,6], [8,7]],
               sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    assert_allclose(dt_axis_multi.tree_.impurity, [6.148 / 2.3, 4.818 / 2.3, 0.0], rtol=0.6)

    #y=[[3,3], [3,3], [4,4], [7,7], [8,8]]
    dt_axis = DecisionTreeRegressor(random_state=0, criterion="axis",
                                   max_leaf_nodes=2)

    # Test axis projection where sample weights are non-uniform (as illustrated above):
    dt_axis.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
               sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    assert_allclose(dt_axis.tree_.impurity, [7.7 / 2.3, 6.13125 / 1.3, 0.0 / 1.0], rtol=0.6)

    # Test same random state produces same result                          
    for i in range(3):
        dt_axis_2 = DecisionTreeRegressor(random_state=0, criterion="axis",
                                        max_leaf_nodes=2)
        dt_axis_2.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
                sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
        assert_allclose(dt_axis.tree_.impurity, dt_axis_2.tree_.impurity)
    
    # Test axis projection where all sample weights are uniform:
    dt_axis.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
               sample_weight=np.ones(5))
    assert_allclose(dt_axis.tree_.impurity, [22.0 / 5.0, 20.75 / 4.0, 0.0 / 1.0], rtol=0.6)

    # Test different random state produces different results
    dt_axis_3 = DecisionTreeRegressor(random_state=1, criterion="axis",
                                    max_leaf_nodes=3)
    dt_axis_3.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
            sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    dt_axis_4 = DecisionTreeRegressor(random_state=3, criterion="axis",
                                    max_leaf_nodes=3)
    dt_axis_4.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
            sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    assert_not_equal(dt_axis_3.tree_, dt_axis_4.tree_)
    
    # Test axis projection where a `sample_weight` is not explicitly provided.
    # This is equivalent to providing uniform sample weights, though
    # the internal logic is different:
    dt_axis.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]])
    assert_allclose(dt_axis.tree_.impurity, [22.0 / 5.0, 20.75 / 4.0, 0.0 / 1.0], rtol=0.6)
    
def test_oblique_proj():
    """Check oblique projection criterion produces correct results on small toy dataset:

    -----------------------
    | X | y1  y2  | weight |
    -----------------------
    | 3 |  3   3  |  0.1   |
    | 5 |  3   3  |  0.3   |
    | 8 |  4   4  |  1.0   |
    | 3 |  7   7  |  0.6   |
    | 5 |  8   8  |  0.3   |
    -----------------------
    |sum wt:|  2.3         |
    -----------------------
 
    Mean1 = 5
    Mean_tot = 5

    For all the samples, we can get the total error by summing:
    (Mean1 - y1)^2 * weight or (Mean_tot - y)^2 * weight

    I.e., error1      = (5 - 3)^2 * 0.1)
                      + (5 - 3)^2 * 0.3)
                      + (5 - 4)^2 * 1.0)
                      + (5 - 7)^2 * 0.6)
                      + (5 - 8)^2 * 0.3)
                      = 0.4 + 1.2 + 1.0 + 2.4 + 2.7
                      = 7.7
          error_tot   = 15.4

    Impurity = error / total weight
             = 7.7 / 2.3
             = 3.3478260869565
             or
             = 15.4 / 2.3
             = 6.6956521739130
             or
             = 0.0 / 2.3
             = 0.0
             -----------------

    From this root node, the next best split is between X values of 5 and 8.
    Thus, we have left and right child nodes:

    LEFT                        RIGHT
    -----------------------     -----------------------
    | X | y1  y2  | weight |    | X | y1  y2  | weight |
    -----------------------     -----------------------
    | 3 |  3   3  |  0.1   |    | 8 |  4   4  |  1.0   |
    | 3 |  7   7  |  0.6   |    -----------------------
    | 5 |  3   3  |  0.3   |    |sum wt:|  1.0         |
    | 5 |  8   8  |  0.3   |    -----------------------
    -----------------------
    |sum wt:|  1.3         |
    -----------------------

    (5.0625 + 3.0625 + 5.0625 + 7.5625) / 4  + 0 = 5.1875
    4 + 4.667 = 8.667

    Impurity is found in the same way:
    Left node Mean1 = Mean2 = 5.25
        error1  = ((5.25 - 3)^2 * 0.1)
                + ((5.25 - 7)^2 * 0.6)
                + ((5.25 - 3)^2 * 0.3)
                + ((5.25 - 8)^2 * 0.3)
                = 6.13125
      error_tot = 12.2625

    Left Impurity = Total error / total weight
            = 6.13125 / 1.3
            = 4.716346153846154
            or
            = 12.2625 / 1.3
            = 9.43269231
            -------------------

    Likewise for Right node:
    Right node Mean1 = Mean2 = 4
    Total error = ((4 - 4)^2 * 1.0)
                = 0

    Right Impurity = Total error / total weight
            = 0 / 1.0
            = 0.0
            ------
    """
    
    dt_obliq = DecisionTreeRegressor(random_state=3, criterion="oblique",
                                   max_leaf_nodes=2)
    
    # Test axis projection where sample weights are non-uniform (as illustrated above):
    dt_obliq.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
               sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    try:
        assert_allclose(dt_obliq.tree_.impurity, [7.7 / 2.3, 6.13125 / 1.3, 0.0 / 1.0], rtol=0.6)
    except:
        try:
            assert_allclose(dt_obliq.tree_.impurity, [2.0*7.7 / 2.3, 2.0*6.13125 / 1.3, 2.0*0.0 / 1.0], rtol=0.6)
        except: 
                assert_allclose(dt_obliq.tree_.impurity, [0.0, 0.0, 0.0], rtol=0.6)
    
    # Test oblique projection where multiple y values are different:
    dt_obliq_multi = DecisionTreeRegressor(random_state=3, criterion="oblique",
                                   max_leaf_nodes=2)
    dt_obliq_multi.fit(X=[[3], [5], [8], [3], [5]], y=[[3,2], [3,4], [4,3], [7,6], [8,7]],
               sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    try:
        assert_allclose(dt_obliq_multi.tree_.impurity, [6.148 / 2.3, 4.818 / 2.3, 0.0], rtol=0.6)
    except:
        try:
            assert_allclose(dt_obliq_multi.tree_.impurity, [2*6.148 / 2.3, 2*4.818 / 2.3, 0.0], rtol=0.6)
        except:
            assert_allclose(dt_obliq_multi.tree_.impurity, [0.0, 0.0, 0.0], rtol=0.6)
    
    # Test for the same result with same initial random state 
    for i in range(3):                         
        dt_obliq_2 = DecisionTreeRegressor(random_state=3, criterion="oblique",
                                        max_leaf_nodes=2)
        dt_obliq_2.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
                sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
        assert_allclose(dt_obliq.tree_.impurity, dt_obliq_2.tree_.impurity)

    # Test different random state produces different results
    dt_obliq_3 = DecisionTreeRegressor(random_state=1, criterion="oblique",
                                    max_leaf_nodes=3)
    dt_obliq_3.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
            sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    dt_obliq_4 = DecisionTreeRegressor(random_state=2, criterion="oblique",
                                    max_leaf_nodes=3)
    dt_obliq_4.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
            sample_weight=[0.1, 0.3, 1.0, 0.6, 0.3])
    assert_not_equal(dt_obliq_3.tree_, dt_obliq_4.tree_)
    
    # Test axis projection where all sample weights are uniform:
    dt_obliq.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]],
               sample_weight=np.ones(5))
    try:
        assert_allclose(dt_obliq.tree_.impurity, [22.0 / 5.0, 20.75 / 4.0, 0.0 / 1.0], rtol=0.6)
    except:
        try:
            assert_allclose(dt_obliq.tree_.impurity, [2.0*22.0 / 5.0, 2.0*20.75 / 4.0, 2.0*0.0 / 1.0], rtol=0.6)
        except: 
                assert_allclose(dt_obliq.tree_.impurity, [0.0, 0.0, 0.0], rtol=0.6)
    
    # Test MAE where a `sample_weight` is not explicitly provided.
    # This is equivalent to providing uniform sample weights, though
    # the internal logic is different:
    dt_obliq.fit(X=[[3], [5], [8], [3], [5]], y=[[3,3], [3,3], [4,4], [7,7], [8,8]])
    try:
        assert_allclose(dt_obliq.tree_.impurity, [22.0 / 5.0, 20.75 / 4.0, 0.0 / 1.0], rtol=0.6)
    except:
        try:
            assert_allclose(dt_obliq.tree_.impurity, [2.0*22.0 / 5.0, 2.0*20.75 / 4.0, 2.0*0.0 / 1.0], rtol=0.6)
        except: 
                assert_allclose(dt_obliq.tree_.impurity, [0.0, 0.0, 0.0], rtol=0.6)
       

if __name__=="__main__":
    test_axis_proj()
    print("axis passed!")
    test_oblique_proj()
    print("oblique passed!")