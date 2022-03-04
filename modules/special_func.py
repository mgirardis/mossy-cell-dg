import numpy
import mpmath

def Gammainc(s,a=0,b=1e324):
    s = numpy.asarray([s]).flatten()
    a = numpy.asarray([a]).flatten()
    b = numpy.asarray([b]).flatten()
    n = numpy.max([len(s),len(a),len(b)])
    s = s*numpy.ones(n)
    a = a*numpy.ones(n)
    b = b*numpy.ones(n)
    G = numpy.frompyfunc(mpmath.gammainc,3,1)
    return numpy.asarray([ mpmath.re(x) if mpmath.im(x)==0.0 else numpy.nan for x in G(s,a,b) ],dtype=float)
