# 2D SIMULATOR CLASSES
from __future__ import print_function
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import csv
import fenics as fe
import numpy as np
import math

# Class Plant. The idea is to create a container for all the parameters linked
# the plant. This means the growth parameters, length and bending stiffness as
# weel as the extension and the rigidification dynamic.
class plant:
    def __init__(self,g,G0,L0,Lg,
                L_final,or_final,reach_final,
                E,n_leav,l_int,
                theta0,kappa0,
                alpha0,beta0,gamma0):
      # Initialization with the parameters
      self.g = g
      self.G0 = G0
      self.L0 = L0
      self.Lg = Lg
      self.L_final = L_final
      self.or_final = or_final
      self.reach_final = reach_final
      self.E = E
      self.n_leav = n_leav
      self.l_int = l_int
      self.theta0 = theta0
      self.kappa0 = kappa0
      self.alpha0 = alpha0
      self.beta0 = beta0
      self.gamma0 = gamma0

      self.read_param('parameters.csv')

    def expressions(self):
        # Expressions definition
        tt = (1/self.G0)*fe.ln(self.Lg/self.L0) # Time when Lg = L(t) if Lg > L0
        # Growth function
        G = fe.Expression('0 <= x[0] && x[0] <= (L0 - (Lg/(exp(t*G)))) ? 0 : G',
                L0 = self.L0, Lg = self.Lg, G = self.G0, t = 0, degree = 0)


        # The following expressions for FR_stem, LD_leav and VD_stem works only
        # if Lg<L0, which is our case. Otherwise, just adapt using the
        # definition of s.
        # I write them in form of expressions for a matter of time of execution:
        # in this way the script is A LOT faster.
        # The multiplication by (36*24*100)**2 is for the conversion of seconds
        # into days of afr, which is the only one that contains time in its
        # unit of measure.

        Rad = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? \
        c + b*(L0 + Lg*G*t - x[0]) + a*pow(L0 + Lg*G*t - x[0],2) : \
        (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
        c + b*((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg))) \
        + a* pow((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg)),2): \
        c + b*((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G)))) \
        + a*pow((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G))),2)',
        a = self.a_rad , b = self.b_rad, c = self.c_rad,
        L0 = self.L0, Lg = self.Lg, G = self.G0,
               t = 0, degree = 1)

        Rad_speed = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? \
        c + b*(L0 + Lg*G*t - x[0]) + a*pow(L0 + Lg*G*t - x[0],2) : \
        (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
        c + b*((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg))) \
        + a* pow((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg)),2): \
        c + b*((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G)))) \
        + a*pow((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G))),2)',
        a = self.a_rsp , b = self.b_rsp, c = self.c_rsp,
        L0 = self.L0, Lg = self.Lg, G = self.G0,
               t = 0, degree = 1)

        FR_stem = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? \
         pow((36*24*100),2)*a/(1 + exp(b*(c -(L0 + Lg*G*t - x[0])))) : \
         (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
         pow((36*24*100),2)*a/(1 + exp(b*(c - \
         (L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg))))) : \
         pow((36*24*100),2)*a/(1 + exp(b*(c - \
         (L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G))))))',
         a = self.a_fr, b = self.b_fr, c = self.c_fr,
         L0 = self.L0, Lg = self.Lg, G = self.G0,
               t = 0, degree = 1)

        # The following is the way in which we inserted the leaves in the simulation
        # We created a step function that at each time t gives a desnity such that
        # once integrated, it gives the total mass of the leaves acting on the point
        # in exam. It is not exactly, but reasonably close to be a localization of the leaves

        LD_leav = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? \
         n_leav*(1/0.13)*a/(1 + exp(b*(c -( L0 + Lg*G*t - div(x[0]*100,13).quot*0.13 )))) : \
         (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
         n_leav*(1/0.13)*a/(1 + exp(b*(c - \
         (L0 + Lg*G*t - div( (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg)*100,13).quot*0.13) ))) : \
         n_leav*(1/0.13)*a/(1 + exp(b*(c - \
         (L0 + Lg*G*t - div((L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G))*100,13).quot*0.13)  )))',
         n_leav = self.n_leav, a = self.a_leav, b = self.b_leav, c = self.c_leav,
         L0 = self.L0, Lg = self.Lg, G = self.G0,
               t = 0, degree = 1)

        VD_stem = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? \
        c + b*(L0 + Lg*G*t - x[0]) + a*pow(L0 + Lg*G*t - x[0],2) : \
        (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
        c + b*((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg))) \
        + a* pow((L0 + Lg*G*t - (Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg)),2): \
        c + b*((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G)))) \
        + a*pow((L0 + Lg*G*t - (L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G))),2)',
        a = self.a_vd , b = self.b_vd, c = self.c_vd,
        L0 = self.L0, Lg = self.Lg, G = self.G0,
               t = 0, degree = 1)

        # Evoluntion of the arc length & derivatives.
        # We must be careful if the initial length is greater or
        # lower than the length of the elongation zone.
        if self.Lg > self.L0:
            s = fe.Expression('0 <= t && t <= tt ? \
                            x[0]*exp(t*G): \
            0 <= et*x[0] && et*x[0] <= Lg - (Lg/(exp((t-tt)*G))) && tt <= t ? \
                       Lg*(log10(Lg/(Lg - et*x[0]))/log10(exp(1))) : \
                       Lg*(1-exp((t-tt)*G)) + Lg*(t-tt)*G + x[0]*exp(t*G)',
                   L0 = self.L0, Lg = self.Lg, G = self.G0, t = 0, tt = tt,
                   et = self.Lg/plant.L0, degree = 1)
            # lbda = ds/dS
            lbda = fe.Expression('0 <= t && t <= tt ? \
                     exp(t*G) : \
            0 <= et*x[0] && et*x[0] <= Lg - (Lg/(exp((t-tt)*G))) && tt <= t ? \
                       Lg*et/(Lg - et*x[0]) : \
                   exp(t*G)',
                   L0 = self.L0, Lg = self.Lg, G = self.G0, t = 0, tt = tt,
                   et = self.Lg/self.L0, degree = 1)
            # s_t = ds/dt
            s_t = fe.Expression('0 <= t && t <= tt ? \
                     G*exp(t*G)*x[0] : \
                 0 <= et*x[0] && et*x[0] <= Lg - (Lg/(exp((t-tt)*G))) ? \
                     0 : \
                   -Lg*G*exp((t-tt)*G) + Lg*G + G*exp(t*G)*x[0]',
                   L0 = self.L0, Lg = self.Lg, G = self.G0, t = 0, tt = tt,
                   et = self.Lg/self.L0, degree = 1)

        else:
            s = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? x[0] : \
                   (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
                       Lg*(log10(Lg/(L0 - x[0]))/log10(exp(1))) + L0 - Lg : \
                   L0*(1-exp(t*G)) + Lg*t*G + x[0]*exp(t*G)',
                   L0 = self.L0, Lg = self.Lg, G = self.G0,
                   t = 0, degree = 1)
            # lbda = ds/dS
            lbda = fe.Expression('0 <= x[0] && x[0] <= (L0 - Lg) ? 1 : \
                   (L0 - Lg) <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? \
                       Lg/(L0-x[0]) : \
                   exp(t*G)',
                   L0 = self.L0, Lg = self.Lg, G = self.G0,
                   t = 0, degree = 1)
            # s_t = ds/dt
            s_t = fe.Expression('0 <= x[0] && x[0] <= L0 - (Lg/(exp(t*G))) ? 0 : \
                   -L0*G*exp(t*G) + Lg*G + G*exp(t*G)*x[0]',
                   L0 = self.L0, Lg = self.Lg, G = self.G0,
                   t = 0, degree = 1)

        return [G,FR_stem,LD_leav,VD_stem,s,lbda,s_t,Rad,Rad_speed]

    def read_param(self,file_name):
        tab = []
        with open(file_name) as csvfile:
            reader = csv.reader(csvfile, delimiter=';',quotechar="'" )
            for row in reader: # each row is a list
                tab.append(row)

        tab = np.array(tab)
        tab = tab[1:,:] # Take away first header row
        row_size = np.size(tab,0)
        col_size = np.size(tab,1)
        tab2 = np.zeros((row_size,col_size)) # If I leave just tab, it doesn't work
        for i in range(row_size):
            for j in range(col_size):
                num = tab[i,j]
                num = num.replace(',','.')
                tab2[i,j] = float(num)

        self.a_fr = tab2[0,0]
        self.b_fr = tab2[0,1]
        self.c_fr = tab2[0,2]
        self.a_vd = tab2[0,3]
        self.b_vd = tab2[0,4]
        self.c_vd = tab2[0,5]
        self.a_leav = tab2[0,6]
        self.b_leav = tab2[0,7]
        self.c_leav = tab2[0,8]
        self.a_rad = tab2[0,9]
        self.b_rad = tab2[0,10]
        self.c_rad = tab2[0,11]
        self.a_rsp = tab2[0,12]
        self.b_rsp = tab2[0,13]
        self.c_rsp = tab2[0,14]

    def LD_stem(self,VD,FR):
        E = self.E
        coeff = E/(4*np.pi*(VD**2))
        return (FR/coeff)**(0.5)

# Class to manage the plant. Compute the evultion of the angle in time and in
# space and consequentely the coordinates in the plane.
class plant_manager:
    def __init__(self,plant,nx,dt,T):
        # The initialization of the class.
        # plant: an instance of the class plant;
        # density: a function. It is used as linear desnity;
        # nx: number of segments in the space discretization;
        # dt: length of the segment in the time discretization;
        # T: final time.
        self.nx = nx
        self.dt = dt
        self.T = T
        self.plant = plant

    def create_mesh(self):
        # Creation of the mesh to solve the dynamic of the system and to
        # reconstruct the coordinates of the stem.
        # V: function space for the dynamic. The domain is a line, the codomain
        #    is in R^4
        # V1: function space for a mesh on a line. The codomain is R.
        #     Used to manage a single function.
        # V_stem: functio  space to reconstruct the stem. The doamin is a line,
        #         the codomain is R^2
        # mesh: the mesh on the segment given of the initial length of the stem.
        mesh = fe.IntervalMesh(self.nx,0,self.plant.L0)
        P1 = fe.FiniteElement('P', fe.interval, 1)
        element = fe.MixedElement([P1, P1, P1, P1])
        V = fe.FunctionSpace(mesh, element)
        V1 = fe.FunctionSpace(mesh, P1)
        element_stem = fe.MixedElement([P1, P1])
        V_stem = fe.FunctionSpace(mesh, element_stem)

        return [V,V1,V_stem,mesh]


    def Variational_Problem(self,functions_eq,t):
        # Definition of the equaitons of the dynamic.
        # functions_eq: the quations used to define the dynamic in the fenics
        #               notation.
        # t: the time which we are considering. Indeed, the equations are
        #    discretized in time.

        # Definitions for convenience of notation
        dt = self.dt
        alpha = self.plant.alpha0
        beta = self.plant.beta0
        gamma = self.plant.gamma0
        E = self.plant.E
        g = self.plant.g
        L0 = self.plant.L0
        Lg = self.plant.Lg
        [G,FR_stem,LD_leav,VD_stem,s,lbda,s_t,Rad,Rad_speed] = self.plant.expressions()
        LD_stem = self.plant.LD_stem
        # Time update
        G.t = t
        FR_stem.t = t - dt
        FR_prev = FR_stem
        FR_stem.t = t
        LD_leav.t = t
        VD_stem.t = t
        lbda.t = t
        s_t.t = t
        Rad.t = t
        Rad_speed.t = t

        u_th = functions_eq[0][0]
        u_ka = functions_eq[0][1]
        u_R  = functions_eq[0][2]
        u_cd = functions_eq[0][3]
        th = functions_eq[1][0]
        ka = functions_eq[1][1]
        R  = functions_eq[1][2]
        cd = functions_eq[1][3]
        th_tr = functions_eq[2][0]
        ka_tr = functions_eq[2][1]
        R_tr  = functions_eq[2][2]
        cd_tr = functions_eq[2][3]
        u    = functions_eq[3]
        u_tr = functions_eq[4]
        th_p = functions_eq[5]
        ka_n = functions_eq[6]

        # Dynamic for Picard
        Fth_p = th_tr.dx(0)*u_th*fe.dx \
            + (1/FR_stem)*lbda*cd_tr*u_th*fe.dx \
            - lbda*ka_tr*u_th*fe.dx

        Fka_p = ka_tr*u_ka*fe.dx - dt*ka_tr.dx(0)*s_t*(1/lbda)*u_ka*fe.dx \
            + dt*(2*Rad_speed/(Rad**2))*G*beta*fe.sin(th_p)*u_ka*fe.dx \
            - dt*(2*Rad_speed/(Rad**2))*G*alpha*fe.cos(th_p)*u_ka*fe.dx \
            + dt*G*gamma*(1/lbda)*th_tr.dx(0)*u_ka*fe.dx \
            - ka_n*u_ka*fe.dx

        FR_p = R_tr.dx(0)*u_R*fe.dx \
            + (LD_stem(VD_stem,FR_stem) + LD_leav)*lbda*u_R*fe.dx

        Fcd_p = cd_tr.dx(0)*u_cd*fe.dx - R_tr*g*fe.sin(th_p)*lbda*u_cd*fe.dx

        F_p = Fth_p + Fka_p + FR_p + Fcd_p
        a_p = fe.lhs(F_p)
        L_p = fe.rhs(F_p)

        # Dynamic for Newton
        Fth_n = th.dx(0)*u_th*fe.dx \
            + (1/FR_stem)*lbda*cd*u_th*fe.dx \
            - lbda*ka*u_th*fe.dx

        Fka_n = ka*u_ka*fe.dx - dt*ka.dx(0)*s_t*(1/lbda)*u_ka*fe.dx \
            + dt*(2*Rad_speed/(Rad**2))*G*beta*fe.sin(th)*u_ka*fe.dx \
            - dt*(2*Rad_speed/(Rad**2))*G*alpha*fe.cos(th)*u_ka*fe.dx \
            + dt*G*gamma*(1/lbda)*th.dx(0)*u_ka*fe.dx \
            - ka_n*u_ka*fe.dx

        FR_n = R.dx(0)*u_R*fe.dx \
            + (LD_stem(VD_stem,FR_stem) + LD_leav)*lbda*u_R*fe.dx

        Fcd_n = cd.dx(0)*u_cd*fe.dx - R*g*fe.sin(th)*lbda*u_cd*fe.dx


        F_n = Fth_n + Fka_n + FR_n + Fcd_n
        Jac  = fe.derivative(F_n, u, u_tr)

        return [a_p, L_p, F_n, Jac]

    def PicardIteration(self,u,th_p,a_p,L_p,bcs,V1,tol,maxiter):
        # Picard Iterations to solve the nonlinear dynamical system.
        # u: function which will comtain the solution of of each iteration
        # th_p: the theta function updated for the picard iteration
        # a_p: bilinear form for the Picard iteration
        #     (th_p becomes a given function)
        # L_p: linear form for the Picard iteration
        # bcs: boundary conditions
        # V1 : function space for a single function
        # tol: tolerance for the Picard iterations
        # maxiter: maximum number of iterations
        eps     = 1.0           # error measure || th - th_p ||
        ite     = 0             # iteration counter
        while eps > tol and ite < maxiter:
            ite += 1
            fe.solve(a_p == L_p, u, bcs)
            th_supp = fe.project(u[0],V1)
            diff    = th_supp.vector().get_local() - th_p.vector().get_local()
            eps     = np.linalg.norm(diff, ord=np.Inf)
            th_p.assign(th_supp)
            # endwhile
        return [u,th_p]

    def NewtonIteration(self,u,F_n,Jac,bcs,tol,maxiter):
        # Newton Iterations  (u from picard as first guess)
        # u: solution of the iteration
        # F_n : function for the Newton iteration
        # Jac: Jacobian of the function F_n
        # bcs: boundary conditions
        # tol: tolerance for the Newton solver
        # maxiter: maximum number of iterations
        problem = fe.NonlinearVariationalProblem(F_n, u, bcs, Jac)
        solver = fe.NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = tol
        prm["newton_solver"]["relative_tolerance"] = tol
        prm["newton_solver"]["maximum_iterations"] = maxiter
        solver.solve()
        return u

    def reconstruction(self,x,mesh,V1):
        # Converts from the fenics format for the functions to a list of values.
        # x: function to convert
        # mesh: the mesh on the segment in consideration
        # V1: function space for the single function
        x_s = fe.project(x,V1)
        coordinates = mesh.coordinates()
        x_vv = x_s.compute_vertex_values(mesh)
        x_co = []
        for i in range(len(coordinates)):
            x_co.append(x_vv[i])
        return x_co

    def stem_coordinates(self,theta,t,V_stem,bcs_stem):
        # Compute the x,y coordinates of the stem starting from the angle
        # theta: function for the angle with respect to the vertical line
        # t: current time
        # V_stem: function space for the functions x and y
        # bcs_stem: boundary conditions for the stem
        # x_s,y_y : functions of the coordinates of the plant in the plane
        u_x, u_y = fe.TestFunctions(V_stem)
        xy_tr = fe.TrialFunction(V_stem)
        x_tr_s, y_tr_s = fe.split(xy_tr)

        [G,FR_stem,LD_leav,VD_stem,s,lbda,s_t,Rad,Rad_speed] = self.plant.expressions()

        lbda.t = t

        F_stem = x_tr_s.dx(0)*u_x*fe.dx - fe.sin(theta)*lbda*u_x*fe.dx + \
                 y_tr_s.dx(0)*u_y*fe.dx - fe.cos(theta)*lbda*u_y*fe.dx

        a_stem = fe.lhs(F_stem)
        L_stem = fe.rhs(F_stem)

        xy = fe.Function(V_stem)
        fe.solve(a_stem == L_stem, xy, bcs_stem)

        x_s, y_s = fe.split(xy)

        return [x_s,y_s]

    def boundary_conditions(self,V,V_stem):
        # Boundary Conditions
        # V: function space for the dynamic
        # V_stem: function space for the coordinates
        # bcs: boundary conditions for the dynamic
        # bcs_stem: boundary conditions for the stem
        def th_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0],0,1E-14)

        def ka_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0],0,1E-14)

        def R_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0],self.plant.L0,1E-14)

        def cd_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0],self.plant.L0,1E-14)

        def xy_boundary(x, on_boundary):
            return on_boundary and fe.near(x[0],0,1E-14)

        th_dir_bc = fe.DirichletBC(V.sub(0), fe.Constant(self.plant.theta0), th_boundary)
        ka_dir_bc = fe.DirichletBC(V.sub(1), fe.Constant(self.plant.kappa0), ka_boundary)
        R_dir_bc  = fe.DirichletBC(V.sub(2), fe.Constant(0), R_boundary)
        cd_dir_bc = fe.DirichletBC(V.sub(3), fe.Constant(0), cd_boundary)

        x_dir_bc = fe.DirichletBC(V_stem.sub(0), fe.Constant(0), xy_boundary)
        y_dir_bc = fe.DirichletBC(V_stem.sub(1), fe.Constant(0), xy_boundary)

        bcs = [th_dir_bc,ka_dir_bc,R_dir_bc,cd_dir_bc]
        bcs_stem = [x_dir_bc, y_dir_bc]

        return [bcs,bcs_stem]

    def solution_computation(self):
        # Computatio of the solution of the dynamic.
        # xall_co: array with the x coordinates in space (rows)
        #          and in tme (columns)
        # yall_co: array with the y coordinates in space (rows)
        #          and in tme (columns)

        # Creation of the function space and mesh
        [V,V1,V_stem,mesh] = self.create_mesh()
        # Test functions for the fem
        u_th, u_ka, u_R, u_cd = fe.TestFunctions(V)
        # Solution
        u = fe.Function(V)
        # Trial functions for fem
        u_tr = fe.TrialFunction(V)
        # Split the functios in their components
        th, ka, R, cd = fe.split(u)
        th_tr, ka_tr, R_tr, cd_tr = fe.split(u_tr)
        # Theta for picard iterations
        th_p = fe.Function(V1)
        # Previous step curvature
        ka_n = fe.project(fe.Constant(self.plant.kappa0),V1)
        # Boundary conditions
        [bcs,bcs_stem] = self.boundary_conditions(V,V_stem)
        # Initialization of the list of coordinates. Actually, thay will
        # be arrays
        xall_co = []
        yall_co = []

        sall_co = []
        Rall_co = []
        VD_stemall_co = []
        LD_leavall_co = []
        kaall_co = []
        # Solution of the dynamic
        t = 0
        while t < self.T:
            t+= self.dt

            functions_eq = [[u_th, u_ka, u_R, u_cd],
                            [th, ka, R, cd],
                            [th_tr, ka_tr, R_tr, cd_tr],
                            u,
                            u_tr,
                            th_p,
                            ka_n]

            F = self.Variational_Problem(functions_eq,t)
            sol_p = self.PicardIteration(u,th_p,F[0],F[1],bcs,V1,1.0E-3,150)
            u = sol_p[0]
            th_p = sol_p[1]

            u = self.NewtonIteration(u,F[2],F[3],bcs,1.0E-5,150)

            th, ka, R, cd = fe.split(u)
            ka_n = fe.project(ka,V1)
            [x_s,y_s] = self.stem_coordinates(th,t,V_stem,bcs_stem)

            [G,FR_stem,LD_leav,VD_stem,s,lbda,s_t,Rad,Rad_speed] = self.plant.expressions()
            s.t = t
            VD_stem.t = t
            LD_leav.t = t

            x_s_list = self.reconstruction(x_s,mesh,V1)
            y_s_list = self.reconstruction(y_s,mesh,V1)

            s_list = self.reconstruction(s,mesh,V1)
            R_s_list = self.reconstruction(R,mesh,V1)
            VD_stem_s_list = self.reconstruction(VD_stem,mesh,V1)
            LD_leav_s_list = self.reconstruction(LD_leav,mesh,V1)
            ka_s_list = self.reconstruction(ka,mesh,V1)

            xall_co.append(x_s_list)
            yall_co.append(y_s_list)

            sall_co.append(s_list)
            Rall_co.append(R_s_list)
            VD_stemall_co.append(VD_stem_s_list)
            LD_leavall_co.append(LD_leav_s_list)
            kaall_co.append(ka_s_list)
        # endwhile

        xall_co = np.transpose(np.array(xall_co))
        yall_co = np.transpose(np.array(yall_co))

        sall_co = np.transpose(np.array(sall_co))
        Rall_co = np.transpose(np.array(Rall_co))
        VD_stemall_co = np.transpose(np.array(VD_stemall_co))
        LD_leavall_co = np.transpose(np.array(LD_leavall_co))
        kaall_co = np.transpose(np.array(kaall_co))

        return [xall_co, yall_co, sall_co, Rall_co,
        LD_leavall_co,VD_stemall_co,kaall_co]

    def compute_reach(self,xall_co,yall_co):
        # Compute the rach of the stem
        # xall_co: array of all the x coordinates of the stem in space and time
        # yall_co: array of all the y coordinates of the stem in space and time
        # reach: computed reach of the stem
        reach = []
        for j in range(np.size(xall_co,1)):
            dist_vector = (xall_co[:,j]**2 + yall_co[:,j]**2)**0.5
            reach = np.append(reach, max(dist_vector))
        return reach

    def compute_orientation(self,xall_co,yall_co):
        # Compute the orientation of the stem
        # xall_co: array of all the x coordinates of the stem in space and time
        # yall_co: array of all the y coordinates of the stem in space and time
        orientation = np.arctan(yall_co[-1,:]/xall_co[-1,:])
        return orientation

    def OPT1(self,s_param):
        self.plant.alpha0 = s_param[0]
        self.plant.beta0 = s_param[1]
        coordinates = self.solution_computation()
        x_final = coordinates[0][-1,-1]
        y_final = coordinates[1][-1,-1]
        orientation = self.plant.or_final
        reach = self.plant.reach_final
        orientation0 = np.arctan(y_final/x_final)
        reach0 = ((x_final)**2 + (y_final)**2)**0.5
        value = abs(orientation - orientation0)**2 + abs(reach - reach0)**2
        return value

    def OPT2(self,dg,gmax,gmin,range_sens):
        param = []
        err = []
        gamma = gmin
        s_param = [self.plant.alpha0,self.plant.beta0]
        amin = s_param[0] - range_sens
        amax = s_param[0] + range_sens
        bmin = s_param[1] - range_sens
        bmax = s_param[1] + range_sens
        while gamma < gmax + dg:
            if gamma == 0:
                gamma = gamma + 0.0001
            self.plant.gamma0 = gamma
            s_param_opt = minimize(self.OPT1,s_param,tol = 1e-2,
            bounds = ((amin,amax),(bmin,bmax)) )
            print(s_param_opt)
            print('Optimal Parameters: ',s_param_opt.x)
            s_param = s_param_opt.x
            param_full = np.append(s_param,gamma)
            value = self.OPT1(s_param)
            param.append(param_full)
            err.append(value)
            gamma = gamma + dg
        return [param,err]

    def OPT3p1(self,s_param):
        self.plant.alpha0 = s_param[0]
        self.plant.beta0 = s_param[1]
        self.plant.gamma0 = s_param[2]
        coordinates = self.solution_computation()
        x_final = coordinates[0][-1,-1]
        y_final = coordinates[1][-1,-1]
        orientation = self.plant.or_final
        reach = self.plant.reach_final
        orientation0 = np.arctan(y_final/x_final)
        reach0 = ((x_final)**2 + (y_final)**2)**0.5
        value = abs(orientation - orientation0)**2 + abs(reach - reach0)**2
        return value

    def OPT3p2(self):
        param = []
        s_param = [self.plant.alpha0,self.plant.beta0,self.plant.gamma0]
        s_param_opt = minimize(self.OPT3p1,s_param,tol = 1e-2,
        bounds = [(-20,20),(-20,20),(-4,4)])
        print(s_param_opt)
        print(s_param_opt.x)
        param = s_param_opt.x
        return [param]
