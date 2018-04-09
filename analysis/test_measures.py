import numpy as np
import pandas as pd
import unittest
from time_series_3d import *

class TestTools(unittest.TestCase):
    """
    Test suite for measurements
    """

    def __init__(self, *args, **kwargs):
        super(TestTools, self).__init__(*args, **kwargs)

    def test_measures(self):
        data=pd.DataFrame({0:[2,4],1:[3,5]})
        s=14.0
        self.assertEqual(n_dot_dot(data),s) # sum table
        for f,g,r in [[n_g_dot,0,5],  # sum first row
                      [n_g_dot,1,9],  # sum second row
                      [n_dot_s,0,6],  # sum first column
                      [n_dot_s,1,8]]: # sum second column
            self.assertEqual(f(data,g),r)
            self.assertEqual(p(data,f,g),r/s)
            self.assertEqual(h(data,f,g),-(r/s)*np.log(r/s))
        # ard
        self.assertEqual(measure_ard(data),
               -(p(data,n_dot_s,0)*np.log(p(data,n_dot_s,0))+ # col 0
                 p(data,n_dot_s,1)*np.log(p(data,n_dot_s,1))  # col 1
               ))
        self.assertEqual(measure_aid(data),
               -(p(data,n_g_dot,0)*np.log(p(data,n_g_dot,0))+ # row 0
                 p(data,n_g_dot,1)*np.log(p(data,n_g_dot,1))  # row 1
               ))
        for f,g,r in [[h_cond_g,0,-(2/5.0*np.log(2/5.0))-(3/5.0*np.log(3/5.0))],
                      [h_cond_g,1,-(4/9.0*np.log(4/9.0))-(5/9.0*np.log(5/9.0))],
                      [h_cond_s,0,-(2/6.0*np.log(2/6.0))-(4/6.0*np.log(4/6.0))],
                      [h_cond_s,1,-(3/8.0*np.log(3/8.0))-(5/8.0*np.log(5/8.0))]]:
            self.assertEqual(f(data,g),r)
        self.assertEqual(measure_wrd(data),h_cond_s_weighted(data,0)+h_cond_s_weighted(data,1))
        self.assertEqual(measure_wid(data),h_cond_g_weighted(data,0)+h_cond_g_weighted(data,1))

        # compare to data from book social foraging theory (pag 245)
        table=pd.DataFrame({"wheat":[144,30,183,118,308,129,131,115,36,91,119,188,379,185,148,16],
                            "corn":[24,45,18,19,16,2,55,39,65,45,12,133,0,17,82,41],
                            "maple_peas":[1,6,1,7,0,1,4,4,0,0,1,1,0,1,4,6],
                            "white_peas":[9,68,19,54,0,5,4,25,0,1,13,0,0,65,29,15],
                            "milo_maize":[182,25,4,9,112,2,339,7,327,109,91,8,24,9,1,98],
                            "vetch":[13,11,0,45,0,7,9,7,4,0,4,0,0,3,7,13],
                            "oats":[2,3,0,0,1,65,2,6,0,7,0,0,3,28,8,8]})
        ard=np.round(measure_ard(table),2)
        self.assertEqual(ard,1.37)
        aid=np.round(measure_aid(table),2)
        self.assertEqual(aid,2.72)
        wrd=np.round(measure_wrd(table),2)
        self.assertEqual(wrd,2.34)
        wid=np.round(measure_wid(table),2)
        self.assertEqual(wid,0.99)

    def test_compute_entropy_measures(self):
        # ARD is low: group specializes to only one resource
        data=pd.DataFrame({0:[50]*100, 1:[0]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(ard,1)        # low: the group consumes resources unevenly
        # ARD is high: individuals generalize
        data=pd.DataFrame({0:[50]*100, 1:[50]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(ard,1)        # high: the group consumes resources evenly
        # ARD is high: individuals specialize
        data=pd.DataFrame({0:[50]*50+[0]*50, 1:[0]*50+[50]*49+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(ard,1)        # high: the group consumes resources evenly
        # WRD is high: individuals have similar (generalized) diets
        data=pd.DataFrame({0:[50]*100, 1:[50]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertGreater(wrd,4)        # high: individuals have similar diets
        # WRD is high: individuals have similar (specialized) diets
        data=pd.DataFrame({0:[50]*100, 1:[0]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertGreater(wrd,4)        # high: individuals have similar diets
        # WRD is low: individuals have different diets
        # data=pd.DataFrame({0:[50]*100, 1:range(100)})
        # ard,wrd,aid,wid=compute_entropy_measures(data)
        # self.assertLess(wrd,2)        # low: individuals have different diets
        # AID is high: individuals generalizes
        data=pd.DataFrame({0:[50]*100, 1:[20]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertGreater(aid,4)        # high: individuals consume the same quantity of resources
        # AID is high: individuals specialize
        data=pd.DataFrame({0:[50]*50+[0]*50, 1:[0]*50+[50]*50})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertGreater(aid,3.9)       # high: individuals consume the same quantity of resources
        # AID is low: some individuals consume more
        data=pd.DataFrame({0:[100]*10+[0]*90, 1:[40]*10+[10]*90})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(aid,4)        # low: some individuals consume more resources
        # WID is high: individuals generalize
        data=pd.DataFrame({0:[50]*100, 1:[50]*99+[1]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertGreater(wrd,0.5)      # high: individuals generalize
        # WID is low: individuals specialize, group generalizes
        data=pd.DataFrame({0:[50]*50+[0]*49+[10], 1:[0]*50+[50]*49+[10]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(wid,0.5)      # low: individuals specialize
        # WID is low: individuals specialize, group specializes
        data=pd.DataFrame({0:[50]*99+[10], 1:[0]*99+[10]})
        ard,wrd,aid,wid=compute_entropy_measures(data)
        self.assertLess(wid,0.5)      # low: individuals specialize
