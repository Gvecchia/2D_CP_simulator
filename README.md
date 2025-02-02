# 2D_CP_simulator
Simulates the growth of a climbing plant stem subject to stimuli and mechanics

Simulations are based on gravitropic/proprioceptive model developed by Bastien,
but contains several additions including the mechanics of the stem. 
The Partial differential equations are solved in time with backward euler method
and in space by using fem method through the fenics package. 

data_extraction.py

reads the .csv file 'congui_pis2_radius' and exprorts the .csv file parameters.

classes.py

contains the classes used in main.py

main.py

simulets the growht of the stem. As input, it requires the parameters.csv file. 
