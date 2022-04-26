from math import pi
import numpy as np
import pygimli as pg
import pybert as pb

try:
    from pygimli import ModellingBase
except:
    from pygimli.core import ModellingBase


# RhoAC = RhoDC * (1-m) ==> dRhoa / m = dRhoa/dRho * dRho/m = JDC * (-RhoDC)
class DCIPMModelling(ModellingBase):
    """ DC/IP modelling class using an (FD-based) approach """
    def __init__(self, f, mesh, rho, verbose=False):
        """ init class with DC forward operator and resistivity vector """
        super().__init__(verbose=verbose)
        self.setMesh(mesh)
        self.f = f
        self.rho = rho  # DC resistivity
        self.mrho = -self.rho
        self.rhoa = f.response(self.rho)
        self.drhoa = -1.0 / self.rhoa
        self.J = pg.matrix.MultLeftRightMatrix(f.jacobian(),
                                               self.drhoa, self.mrho)
        self.setJacobian(self.J)

    def response(self, m):
        """ return forward response as function of chargeability model """
        rho = self.rho * (1. - m)
        if self.verbose:
            print('min/max m=', min(m), max(m), end=" ")
            print('min/max rho=', min(rho), max(rho), end=" ")
        ma = 1.0 - self.f.response(rho) / self.rhoa
        if self.verbose:
            print('min/max ma=', min(ma), max(ma))
        return ma + 1e-4

    def createJacobian(self, model):
        """ create jacobian matrix using unchanged DC jacobian and m model """
        pass  # do nothing (but prevent brute-force jacobian calculation)
#        self.J.left = - model * 1.0  # prevent reference change


class ERTTLmod(ModellingBase):
    """ ERT timelapse modelling class based on BlockMatrices """
    def __init__(self, nf=0, data=None, mesh=None, fop=None, rotate=False,
                 set1back=True, verbose=False):
        """ Parameters: """
        super(type(self), self).__init__(verbose)
        if fop is not None:
            data = fop.data()
            mesh = fop.mesh()
        if type(data) == list:  # list of data with different size
            nf = len(data)

        self.nf = nf
        self.nd = []
        self.FOP2d = []
        for i in range(nf):
            if type(data) is list:
                fopi = pb.DCSRMultiElectrodeModelling(mesh, data[i])
            else:
                fopi = pb.DCSRMultiElectrodeModelling(mesh, data)
            self.nd.append(fopi.data().size())
            if set1back:
                fopi.region(1).setBackground(True)
            fopi.createRefinedForwardMesh(True)
            self.FOP2d.append(fopi)

        self.id = np.hstack((0, np.cumsum(self.nd, dtype=np.int64)))  # indices
        self.pd2d = pg.Mesh(self.FOP2d[0].regionManager().paraDomain())
        print("2D PD:", self.pd2d)
        for c in self.pd2d.cells():
            c.setMarker(0)

        self.nc = self.pd2d.cellCount()
        self.J = pg.core.RBlockMatrix()
        for i in range(nf):
            n = self.J.addMatrix(self.FOP2d[i].jacobian())
            self.J.addMatrixEntry(n, int(self.id[i]), self.nc*i)

        z = pg.Vector(range(nf+1))
        self.mesh3D = pg.core.createMesh3D(self.pd2d, z, 0, 0)
        if rotate:
            self.mesh3D.swapCoordinates(1,2)
            # self.mesh3D.rotate(pg.Pos(pi/2, 0, 0))

        self.mesh3D.exportVTK('mesh3d.vtk')
        self.setMesh(self.mesh3D)
        print("3D PD:", self.mesh3D)
        if 0:  # maybe skip it so that it will force recalc in 1st iteration
            self.FOP2d[0].jacobian().resize(data.size(), self.nc)
            self.FOP2d[-1].jacobian().resize(data.size(), self.nc)
            self.J.recalcMatrixSize()
            print(self.J.rows(), self.J.cols())

        self.setJacobian(self.J)

    def response(self, model):
        """ cut-together forward responses of all soundings """
        resp = pg.Vector(int(self.id[-1]))
        for i in range(self.nf):
            resp.setVal(self.FOP2d[i].response(
                model[self.nc*i:self.nc*(i+1)]),
                int(self.id[i]), int(self.id[i+1]))

        return resp

    def createJacobian(self, model):
        """Compute Jacobian matrix by individual (block) Jacobians."""
        for i in range(self.nf):
            self.FOP2d[i].createJacobian(model[self.nc*i:self.nc*(i+1)])


class ERTMultiPhimod(ModellingBase):
    """ FDEM 2d-LCI modelling class based on BlockMatrices """
    def __init__(self, pd, J2d, nf, rotate=False, verbose=False):
        """ Parameters: FDEM data class and number of layers """
        super(ERTMultiPhimod, self).__init__(verbose)
        self.nf = nf
        self.mesh2d = pd
        self.pd2d = pd
        self.nc = pd.cellCount()
        for c in self.pd2d.cells():
            c.setMarker(0)
        self.nd = J2d.rows()
        self.J2d = J2d
        self.FOP2d = pg.core.LinearModelling(pd, J2d)
        self.J = pg.core.RBlockMatrix()
        for i in range(self.nf):
            n = self.J.addMatrix(self.J2d)
            self.J.addMatrixEntry(n, self.nd*i, self.nc*i)

        z = pg.Vector(range(nf+1))
        self.mesh3D = pg.core.createMesh3D(self.pd2d, z, 0, 0)
        if rotate:
            self.mesh3D.swapCoordinates(1, 2)  # interchange y/z for zWeight
            # self.mesh3D.rotate(pg.Pos(pi/2, 0, 0))

        self.setMesh(self.mesh3D)
        print(self.nd*self.nf, self.nc*self.nf, self.mesh3D.cellCount())
        self.J.recalcMatrixSize()
        print(self.J.rows(), self.J.cols())
        self.setJacobian(self.J)

    def response(self, model):
        """ cut-together forward responses of all soundings """
        resp = pg.Vector(self.nd*self.nf)
        for i in range(self.nf):
            modeli = model[self.nc*i:self.nc*(i+1)]
            resp.setVal(self.J2d * modeli,
                        self.nd*i, self.nd*(i+1))

        return resp

    def createJacobian(self, model):
        pass
