# sal's custom deinterlace and IVTC functions
# vapoursynth plugins are in /usr/lib/x86_64-linux-gnu/vapoursynth

import vapoursynth as vs

import functools
import math
import fractions as fr
import saltools

core = vs.get_core()

# saltools.combprotect

def StatsNewLine():
	return null

def NoiseLevelDo(n,f,clip,core, statsfile):
	noisestr = "{!r}\tcomplexity=\t{!r}\n".format(n,f.props._Average)
	statsfile.write(noisestr)
	return clip

def NoiseStatCopy(n, f):
   fout = f[0].copy()
   fout.props._Average = f[1].props._Average
   return fout

def NoiseLevel(clip, statsfile):

	sqr = core.std.SeparateFields(clip, tff=1).std.SelectEvery(2,0).resize.Lanczos(512, 512, filter_param_a=16,format=vs.YUV420P10) #16 taps should allow all pixels in a DCI 4k frame to be considered in a 512 wide picture.
	sqr = core.convo2d.Convolution(sqr, matrix=[-1,-2,-1,-2,32,-2,-1,-2,-1])
	#sqr=core.std.SeparateFields(clip, tff=1).std.SelectEvery(2,0).resize.Lanczos(width=768,height=768, format=vs.YUV420P10)
	blk=8
	ovr=0
	sad=400

	sqrspr = core.mv.Super(sqr,pel=2)
	sqrvf = core.mv.Analyse(sqrspr,blksize=blk,overlap=ovr,dct=False,isb=False)
	sqrcmpf = core.mv.Compensate(sqr,sqrspr,sqrvf,thsad=sad,thscd1=300,thscd2=128)
	sqrvb = core.mv.Analyse(sqrspr,blksize=blk,overlap=ovr,dct=False,isb=True)
	sqrcmpb = core.mv.Compensate(sqr,sqrspr,sqrvf,thsad=sad,thscd1=300,thscd2=128)
	# "x 10 * 1 + log 10 log /"
	noise = core.std.Expr([sqr,sqrcmpf,sqrcmpb], expr=["x y z + 2 / - abs","512","512"])
	noise = core.std.PlaneStats(noise, plane=0, prop='_')
	stats = open(statsfile,'a',1)
	
	getnoise = core.std.FrameEval(noise, functools.partial(NoiseLevelDo, clip=noise, core=core, statsfile=stats), prop_src=noise)
	return core.std.ModifyFrame(clip=clip, clips=[clip, getnoise], selector=NoiseStatCopy)

def FieldFindDo(n,f,clip,core, statsfile):
	noisestr = "{!r}\tcomplexity=\t{!r}\n".format(n,f.props._MinMax[1])
	statsfile.write(noisestr)
	return clip

def FieldFindCopy(n, f):
	fout = f[0].copy()
	fout.props._MinMax = f[1].props._MinMax
	return fout

def FieldFind(clip, statsfile, show=False):
	# 115 and above is a good place to look for interlace. currently dependent on brightness.
	# 115 is a low point on a calm scene from blurry standards-converted interlaced SD.
	# progressive content showing above this will be very sharp or noisy.
	sqr = core.resize.Bilinear(clip=clip,width=512,format=vs.GRAY16)#.YUV420P8)
	b3 = core.std.SeparateFields(sqr, tff=1).convo2d.Convolution([1,2,1], mode='v').std.DoubleWeave(tff=1).std.SelectEvery(2,0)
	b5 = core.convo2d.Convolution(sqr, [6,9,6,9,6], mode='v')
	noise = core.std.Expr(clips=[b3,b5], expr=["x y - abs"])
	noise = core.std.PlaneStats(noise, plane=0, prop='_')
	stats = open(statsfile,'a',1)

	getnoise = core.std.FrameEval(noise, functools.partial(FieldFindDo, clip=noise, core=core, statsfile=stats), prop_src=noise)
	if show:
		return core.std.ModifyFrame(clip=noise, clips=[noise, getnoise], selector=FieldFindCopy)
	else:
		return core.std.ModifyFrame(clip=clip, clips=[clip, getnoise], selector=FieldFindCopy)