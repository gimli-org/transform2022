"""Tools for (time domain) IP modelling using chargeability."""
import pygimli as pg
import numpy as np
import matplotlib.pyplot as plt
from pygimli.utils import rmswitherr
from scipy.special import gamma

try:
    from pygimli import ModellingBase
except ImportError:
    from pygimli.core import ModellingBase


class IPSeigelModelling(ModellingBase):
    """DC/IP modelling class using an (FD-based) approach."""

    def __init__(self, f, mesh, rho, verbose=False, response=None):
        """Init class with DC forward operator and resistivity vector."""
        super().__init__(verbose=verbose)
        self.setMesh(mesh)
        self.f = f
        self.rhoDC = rho  # DC resistivity
        self.drhodm = -self.rhoDC
        if response is None:
            response = f.response(self.rhoDC)

        self.rhoaDC = response

        self.dmdrhoa = -1.0 / self.rhoaDC
        self.J = pg.matrix.MultLeftRightMatrix(f.jacobian(), self.dmdrhoa,
                                               self.drhodm)
        self.setJacobian(self.J)
        self.fullJacobian = False

    def response(self, m):
        """Return forward response as function of chargeability model."""
        self.rhoaAC = self.f.response(self.rhoDC / (1. - m))
        if self.fullJacobian:
            self.dmdrhoa = -1.0 / self.rhoaAC

        return pg.abs(1.0 - self.rhoaDC / self.rhoAC)

    # RhoAC = RhoDC * (1-m) => dRhoa / m = dRhoa/dRho * dRho/m = JDC * (-RhoDC)
    def createJacobian(self, model):
        """Create jacobian matrix using unchanged DC jacobian and m model."""
        if self.fullJacobian:
            self.rhoAC = self.rhoDC * (1 - model)
            self.J.right = -self.rhoAC
            self.f.createJacobian(self.rhoAC)


class DCIPSeigelModelling(ModellingBase):
    """DC/IP modelling class using an (FD-based) approach."""

    # def __init__(self, f=None, mesh=None, rho=None, resp=None, verbose=False):
    def __init__(self, ERT, verbose=False):
        """Init class with DC forward operator and resistivity vector."""
        super().__init__(verbose=verbose)
        self.setMesh(ERT.paraDomain)
        self.resp = ERT.inv.response
        self.res = ERT.inv.model
        self.J = pg.matrix.MultLeftRightMatrix(ERT.fop.jacobian(),
                                               1./self.resp, self.res)
        # G = pg.utils.gmat2numpy(ERT.fop.jacobian())
        # Glog = np.reshape(1./self.resp, (-1, 1)) * G * self.res
        # self.J = Glog / np.sum(Glog, 1)
        self.setJacobian(self.J)

    def response(self, m):
        """Return forward response as function of chargeability model."""
        return self.J.dot(m)

    def createJacobian(self, model):
        """Create jacobian matrix using unchanged DC jacobian and m model."""
        pass




class DCIPMModelling(ModellingBase):
    """DC/IP modelling class using an (FD-based) approach."""

    def __init__(self, f, mesh, rho, verbose=False, response=None):
        """Init class with DC forward operator and resistivity vector."""
        super().__init__(verbose=verbose)
        self.setMesh(mesh)
        self.f = f
        self.rhoDC = rho  # DC resistivity
        self.drhodm = -self.rhoDC
        if response is None:
            self.rhoaDC = f.response(self.rhoDC)
        else:
            self.rhoaDC = response

        self.dmdrhoa = -1.0 / self.rhoaDC
        self.J = pg.matrix.MultLeftRightMatrix(f.jacobian(), self.dmdrhoa,
                                               self.drhodm)
        self.setJacobian(self.J)
        self.fullJacobian = False

    def response(self, m):
        """Return forward response as function of chargeability model."""
        self.rhoaAC = self.f.response(self.rhoDC * (1. - m))
        if self.fullJacobian:
            self.dmdrhoa = -1.0 / self.rhoaAC

        return pg.abs(1.0 - self.rhoaAC / self.rhoaDC)

    # RhoAC = RhoDC * (1-m) => dRhoa / m = dRhoa/dRho * dRho/m = JDC * (-RhoDC)
    def createJacobian(self, model):
        """Create jacobian matrix using unchanged DC jacobian and m model."""
        if self.fullJacobian:
            self.rhoAC = self.rhoDC * (1 - model)
            self.J.right = -self.rhoAC
            self.f.createJacobian(self.rhoAC)


class DCIPSeigelModelling(ModellingBase):
    """DC/IP modelling class using an (FD-based) approach."""

    # def __init__(self, f=None, mesh=None, rho=None, resp=None, verbose=False):
    def __init__(self, ERT, verbose=False):
        """Init class with DC forward operator and resistivity vector."""
        super().__init__(verbose=verbose)
        self.setMesh(ERT.paraDomain)
        self.resp = ERT.inv.response
        self.res = ERT.inv.model
        self.J = pg.matrix.MultLeftRightMatrix(ERT.fop.jacobian(),
                                               1./self.resp, self.res)
        # G = pg.utils.gmat2numpy(ERT.fop.jacobian())
        # Glog = np.reshape(1./self.resp, (-1, 1)) * G * self.res
        # self.J = Glog / np.sum(Glog, 1)
        self.setJacobian(self.J)

    def response(self, m):
        """Return forward response as function of chargeability model."""
        return self.J.dot(m)

    def createJacobian(self, model):
        """Create jacobian matrix using unchanged DC jacobian and m model."""
        pass


class DCIPMSmoothModelling(ModellingBase):
    """Simultaneous IP modelling of several gates."""

    def __init__(self, f, mesh, rho, t, verbose=False):
        """Init class with DC forward operator and resistivity vector."""
        super().__init__(verbose=verbose)
        self.nt = len(t)
        self.nc = mesh.cellCount()
        self.nd = f.data.size()
        self.mesh3d = pg.core.createMesh3D(mesh, np.arange(self.nt+1))
        self.mesh3d.swapCoordinates(1, 2)
        for c in self.mesh3d.cells():
            c.setMarker(0)
        self.setMesh(self.mesh3d)
        self.fops = [DCIPMModelling(f, mesh, rho) for i in range(self.nt)]
        self.J = pg.core.RBlockMatrix()
        for i in range(self.nt):
            n = self.J.addMatrix(self.fops[i].jacobian())
            self.J.addMatrixEntry(n, int(self.nd*i), self.nc*i)

        self.setJacobian(self.J)

    def response(self, m):
        """Model response (concatenated model responses)."""
        mm = np.reshape(m, (-1, self.nc))
        return np.hstack([self.fops[i].response(mm[i])
                          for i in range(self.nt)])

    def createJacobian(self, m):
        """Create Jacobian by calling individual jacobians."""
        mm = np.reshape(m, (-1, self.nc))
        for i, mi in enumerate(mm):
            self.fops[i].createJacobian(mi)


def ColeColeTD(t, m=1.0, tau=1.0, c=0.5, jmax=None, eps=1e-5):
    """Cole-Cole time-domain response."""
    u = np.zeros_like(t)
    if jmax is None:
        jmax = int(10/c) * 30
    for j in range(jmax):
        du = (-1)**j * (t/tau)**(j*c) / gamma(1+j*c)
        u += du
        if np.linalg.norm(du)/np.linalg.norm(u) < eps:
            break

    # print(j, jmax, np.linalg.norm(du)/np.linalg.norm(u))
    return u * m


class CCTDModelling(ModellingBase):
    """Cole-Cole time domain modelling class."""

    def __init__(self, tvec, verbose=False):
        """Initialize model with time vector."""
        pg.core.ModellingBase.__init__(self, verbose)
        self.tvec = tvec
        self.setMesh(pg.meshtools.createMesh1D(1, 3))

    def response(self, model):
        """Return Cole-Cole time domain forward response."""
        return ColeColeTD(self.tvec, *model)


if __name__ == "__main__":
    zw = 0.5
    dc = Resistivity('Onneslov.data')
    print("DC min/max = ", min(dc.data('ip')), max(dc.data('ip')))
    dc.setMeshPot()
    dc.fop.setThreadCount(8)
    if 0:
        dc.invert(zweight=zw)  # , recalcJacobian=False, maxIter=2)
        dc.showResultAndFit(cMin=0.5, cMax=500)
        dc.figs['resultFit'].savefig('resultFit.pdf')
        dc.mesh.save('dcmesh.bms')
        dc.resistivity.save('res.vec')
    else:
        dc.resistivity = pg.Vector('resistivity.vector')
        dc.response = dc.fop(dc.resistivity)
        dc.fop.createJacobian(dc.resistivity)
        dc.paraDomain = dc.fop.regionManager().paraDomain()
    # %%
    # iperr = np.array(dc.error)
    ma = dc.data('ip') * 1e-3  # mrad to rad
    iperr = pg.Vector(dc.data.size(), 0.01)
    if 1:
        mmin, mmax = 1e-3, 0.2
        print('discarding min/max', sum(ma < mmin), sum(ma > mmax))
        iperr[ma < mmin] = 1e5
        ma[ma < mmin] = mmin
        iperr[ma > mmax] = 1e5
        ma[ma > mmax] = mmax
    # %% set up forward operator
    fIP = DCIPMModelling(dc.fop, dc.mesh, dc.resistivity)
    fIP.region(1).setBackground(True)
    fIP.region(2).setConstraintType(1)
    fIP.region(2).setZWeight(zw)
    fIP.createRefinedForwardMesh(True)
    tD, tM = pg.trans.TransLog(), pg.trans.TransLogLU(0, 0.95)
    INV = pg.core.RInversion(ma, fIP, tD, tM, True, False)
    mstart = pg.Vector(len(dc.resistivity), pg.median(ma))  # 10 mV/V
    INV.setModel(mstart)
    INV.setAbsoluteError(iperr)
    INV.setLambda(10)
    INV.setRobustData(True)
    dc.m = INV.run()
    dc.m.save("m.vec")
    dc.paraDomain = fIP.regionManager().paraDomain()
    cMin, cMax = 10, 300  # max(ma)*1e3
    # %%
    respma = INV.response()  # AC/DC dc.data('rhoa')
    print(max(ma), max(respma), max(dc.m))
    myrms = rmswitherr(ma, respma, INV.error(), 0.4)
    print("RMS=", myrms)
    fig, ax = plt.subplots(nrows=3, figsize=(8, 12))
    # dc.showModel(ax=ax[2], vals=dc.m*1e3, cMin=cMin, cMax=cMax, logScale=False)
    pg.show(dc.paraDomain, dc.m*1e3, ax=ax[2], cMin=cMin, cMax=cMax,
            logScale=True, colorBar=True)
    for i, mdata in enumerate([ma, respma]):
        dc.showData(ax=ax[i], vals=mdata*1e3, cMin=cMin, cMax=cMax, logScale=False,
                    colorBar=False)
    # %%
    fig.savefig('outMaM.pdf', bbox_inches='tight')



