# sal's custom deinterlace and IVTC functions
# vapoursynth plugins are in /usr/lib/x86_64-linux-gnu/vapoursynth

import vapoursynth as vs

import functools
import math
import numpy as np

core = vs.get_core()

def letterbox(clip, left=0, right=0, top=0, bottom=0):
	import math
	left=math.floor(left)	
	right=math.floor(right)
	top=math.floor(top)
	bottom=math.floor(bottom)
	if min([left,right,top,bottom]) >= 0:
		v1 = clip.std.CropRel(left, right, top, bottom).std.AddBorders(left, right, top, bottom)
	else:
		v1 = clip
	return v1

def pad(clip, w=1920, h=1080):
	import math
	left = math.floor((w - clip.width)/2)
	right = (w - clip.width) - left
	top = math.floor((h - clip.height)/2)
	bottom = (h - clip.height) - top
	return clip.std.AddBorders(left, right, top, bottom)

def squarepix(clip, height=576, crop=1.0, aspect=16/9, outaspect=16/9, sd_mode=702,off_w=0,off_h=0, scale=1):
	# if crop is 0, full pillarbox. if crop is 1, full crop. anything in between is a compromise.
	# you should handle deinterlacing before running me.
	# express aspect as a fraction or float
	# sd_mode determines if we're working in 702 maths (analog, the "correct" way), or 720 or whatever.
	# hq does a nnedi3 upscale first. overkill for archival SD, worth doing for sharp stuff.
	clip = clip.fmtc.resample(css="444")
	width = int(outaspect*height)
	inaspect=aspect
	par=outaspect/inaspect
	sw = (((sd_mode*par)*(1-crop))+(sd_mode*(crop)))/scale
	sh = (((clip.height/par)*(crop))+(clip.height*(1-crop)))/scale
	spadx = (clip.width-sw)/2
	spady = (clip.height-sh)/2
	opad = -spadx * width/sw
	v1 = clip.fmtc.resample(w=width,h=height,sw=sw,sh=sh,sx=spadx-off_w,sy=spady+off_h)
	v1 = letterbox(v1,opad,opad)
	return v1

def SDtoHD(clip, crop=1.0, aspect=16/9, sd_mode=702, hq=True,off_w=0,off_h=0, scale=1, hd720=False):
	# if crop is 0, full pillarbox. if crop is 1, full crop. anything in between is a compromise.
	# you should handle deinterlacing before running me.
	# express aspect as a fraction or float
	# sd_mode determines if we're working in 702 maths (analog, the "correct" way), or 720 or whatever.
	# hq does a nnedi3 upscale first. overkill for archival SD, worth doing for sharp stuff.
	if hd720:
		width=1280
		height=720
	else:
		width=1920
		height=1080

	clip = clip.fmtc.resample(css="444")	
	if hq:
		sd_mode = sd_mode*2
		clip = clip.std.Transpose().nnedi3.nnedi3(0,dh=True).std.Transpose().nnedi3.nnedi3(0,dh=True).fmtc.resample(sx=.5,sy=.5,css="444")
	outaspect=16/9
	inaspect=aspect
	par=outaspect/inaspect
	sw = (((sd_mode*par)*(1-crop))+(sd_mode*(crop)))/scale
	sh = (((clip.height/par)*(crop))+(clip.height*(1-crop)))/scale
	spadx = (clip.width-sw)/2
	spady = (clip.height-sh)/2
	opad = -spadx * width/sw
	v1 = clip.fmtc.resample(w=width,h=height,sw=sw,sh=sh,sx=spadx-off_w,sy=spady+off_h)
	v1 = letterbox(v1,opad,opad)
	return v1

def combprotect(clip):
	soft =       [1, 2, 1,
   		          0, 8, 0,
            		  1, 2, 1]
	shrp =       [-1, -2, -1,
 	               0, 17,  0,
	              -1, -2, -1]	
	
	return clip.convo2d.Convolution(soft).convo2d.Convolution(shrp)

def combprotect2(clip, fields=False):
	soft5 =      [1, 1, 1, 1, 1]

	soft3 =      [1, 1, 1]

	if fields:
		return clip.convo2d.Convolution(soft3,mode='v')
	else:
		return clip.convo2d.Convolution(soft5,mode='v')


def conditionalDeint(n, f, orig, deint):
    if f.props._Combed:
        return deint
    else:
        return orig

def fieldmatchproc(n, f, clip, swapped):
	core = vs.get_core()
	if f[0].props.sameAverage > f[1].props.swapAverage:
		swappeddeint = core.tdm.TDeintMod(swapped, order=1, edeint=core.nnedi3.nnedi3(swapped, field=1))
		swappedprop = core.tdm.IsCombed(swapped)
		swapped = core.std.FrameEval(swapped, functools.partial(conditionalDeint, orig=swapped, deint=swappeddeint), swappedprop)
		return swapped
	
	else:
		clipdeint = core.tdm.TDeintMod(clip, order=0, edeint=core.nnedi3.nnedi3(clip, field=0))
		clipprop = core.tdm.IsCombed(clip)
		clip = core.std.FrameEval(clip, functools.partial(conditionalDeint, orig=clip, deint=clipdeint), clipprop)
		return clip

def fieldmatch(clip):
	core = vs.get_core()
	comb =		[-1, 2, -1]
	swapped = core.std.SeparateFields(clip, tff=True).std.DoubleWeave(tff=True).std.SelectEvery(2, 1)
	swapmetric = core.convo2d.Convolution(swapped, comb, mode="v")
	swapstat = core.std.PlaneStats(swapmetric, plane=0, prop='swap')
	samemetric = core.convo2d.Convolution(clip, comb, mode="v")
	samestat = core.std.PlaneStats(samemetric, plane=0, prop='same')
	clip = core.std.FrameEval(clip, functools.partial(fieldmatchproc,clip=clip,swapped=swapped), prop_src=[samestat,swapstat])
	
	deint = core.tdm.TDeintMod(clip, order=1, edeint=core.nnedi3.nnedi3(clip, field=0))
	combProps = core.tdm.IsCombed(clip)
	clip = core.std.FrameEval(clip, functools.partial(conditionalDeint, orig=clip, deint=deint), combProps)	
	
	return combprotect(clip)

def deinterlace(clip, dbl=False, order=1, usefield="noodle"):
	# order = 0 for bff, 1 for tff. will strip out current order property of clip.
	if usefield == "noodle":
		usefield = order
	core = vs.get_core()
	nnediorder = order
	yadmode = 0
	if dbl:
		nnediorder += 2
		yadmode = 1
	clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=order+1) # 1 = bff, 2 = tff
	edeint = core.nnedi3.nnedi3(clip,nnediorder,nsize=4,nns=0)
	#v1 = core.yadifmod.Yadifmod(clip=clip,edeint=edeint,order=order,field=usefield,mode=yadmode)
	v1 = core.tdm.TDeintMod(clip, order=order, field=usefield, mode=yadmode, length=10, mtype=2, ttype=5, nt=2, edeint=edeint)
	return combprotect(v1)

# Sal's custom NR FUNctions.

def dvnr(clip, sad=300, scd1=75, scd2=50, blk=8, pel=2):
	core = vs.get_core()
	scd2=int(255*(scd2/100))

	v1 = clip
	vsuper = core.mv.Super(v1,pel=pel)
	vf = core.mv.Analyse(vsuper,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	#core.mv.Compensate(clip, super, vectors, scbehavior, thsad, fields, thscd1, thscd2, isse, tff)
	compf = core.mv.Compensate(v1,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb = core.mv.Analyse(vsuper,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compb = core.mv.Compensate(v1,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	interleaved = core.std.Interleave([compf,v1,compb], mismatch=False)
	#degrained = core.dgm.DegrainMedian(interleaved, 16, mode=1)
	degrained = core.misc.AverageFrames(interleaved,[1,1,1])
	v1 = core.std.SelectEvery(degrained, 3, 1)
	return v1

def dvnr3(clip, sad=390, scd1=200, scd2=50, blk=8, pel=4):
	core = vs.get_core()
	scd2=int(255*(scd2/100))

	v1 = clip
	blur = core.bilateral.Bilateral(v1, ref=v1, sigmaS=6*v1.height/1080, sigmaR=.025, algorithm=2,planes=[0])
	blur = core.bilateral.Bilateral(blur, ref=blur, sigmaS=16*v1.height/1080, sigmaR=.025, algorithm=2,planes=[1,2])
	vsuper = core.mv.Super(v1,pel=pel)
	vsuperblur = core.mv.Super(blur,pel=pel)
	vf = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	#core.mv.Compensate(clip, super, vectors, scbehavior, thsad, fields, thscd1, thscd2, isse, tff)
	compf = core.mv.Compensate(v1,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compb = core.mv.Compensate(v1,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	interleaved = core.std.Interleave([compf,v1,compb], mismatch=False)
	#degrained = core.dgm.DegrainMedian(interleaved, 16, mode=1)
	degrained = core.misc.AverageFrames(interleaved,[1,1,1])
	v1 = core.std.SelectEvery(degrained, 3, 1)
	return v1

def dvnr5(clip, sad=390, scd1=200, scd2=50, blk=16, pel=4):
	core = vs.get_core()
	scd2=int(255*(scd2/100))

	v1 = clip
	blur = core.bilateral.Bilateral(v1, ref=v1, sigmaS=6*v1.height/1080, sigmaR=.025, algorithm=2,planes=[0])
	blur = core.bilateral.Bilateral(blur, ref=blur, sigmaS=16*v1.height/1080, sigmaR=.025, algorithm=2,planes=[1,2])
	vsuper = core.mv.Super(v1,pel=pel)
	vsuperblur = core.mv.Super(blur,pel=pel)
	vf = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	#core.mv.Compensate(clip, super, vectors, scbehavior, thsad, fields, thscd1, thscd2, isse, tff)
	compf = core.mv.Compensate(v1,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compb = core.mv.Compensate(v1,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	vf2 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False,delta=2)
	#core.mv.Compensate(clip, super, vectors, scbehavior, thsad, fields, thscd1, thscd2, isse, tff)
	compf2 = core.mv.Compensate(v1,vsuper,vf2,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb2 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True,delta=2)
	compb2 = core.mv.Compensate(v1,vsuper,vb2,thsad=sad,thscd1=scd1,thscd2=scd2)
	interleaved = core.std.Interleave([compf2,compf,v1,compb,compb2], mismatch=False)
	#degrained = core.dgm.DegrainMedian(interleaved, 16, mode=1)
	degrained = core.dgm.DegrainMedian(interleaved, 255, mode=4).misc.AverageFrames([1,1,1,1,1])
	v1 = core.std.SelectEvery(degrained, 5, 2)
	return v1

def dvnr7(clip, sad=250, scd1=200, scd2=50, blk=16, pel=4):
	core = vs.get_core()
	scd2=int(255*(scd2/100))

	v1 = clip
	blur = core.bilateral.Bilateral(v1, ref=v1, sigmaS=6*v1.height/1080, sigmaR=.025, algorithm=2,planes=[0])
	blur = core.bilateral.Bilateral(blur, ref=blur, sigmaS=16*v1.height/1080, sigmaR=.025, algorithm=2,planes=[1,2])
	vsuper = core.mv.Super(v1,pel=pel)
	vsuperblur = core.mv.Super(blur,pel=pel)
	vf = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	compf = core.mv.Compensate(v1,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compb = core.mv.Compensate(v1,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	vf2 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False,delta=2)
	compf2 = core.mv.Compensate(v1,vsuper,vf2,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb2 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True,delta=2)
	compb2 = core.mv.Compensate(v1,vsuper,vb2,thsad=sad,thscd1=scd1,thscd2=scd2)
	vf3 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False,delta=3)
	compf3 = core.mv.Compensate(v1,vsuper,vf3,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb3 = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True,delta=3)
	compb3 = core.mv.Compensate(v1,vsuper,vb3,thsad=sad,thscd1=scd1,thscd2=scd2)
	interleaved = core.std.Interleave([compf3,compf2,compf,v1,compb,compb2,compb3], mismatch=False)
	degrained = core.dgm.DegrainMedian(interleaved, 255, mode=4).misc.AverageFrames([1,1,1,1,1,1,1])
	v1 = core.std.SelectEvery(degrained, 7, 3)
	return v1

def RemoveDirt(input, repmode=16):
	core = vs.get_core()
	cleansed = core.rgvs.Clense(input)
	sbegin = core.rgvs.ForwardClense(input)
	send = core.rgvs.BackwardClense(input)
	scenechange = core.rgvs.SCSelect(input, sbegin, send, cleansed)
	alt = core.rgvs.Repair(scenechange, input, mode=[repmode,repmode,1])
	restore = core.rgvs.Repair(cleansed, input, mode=[repmode,repmode,1])
	corrected = core.rgvs.RestoreMotionBlocks(cleansed, restore, neighbour=input, alternative=alt, gmthreshold=70, dist=1, dmode=2, noise=10, noisy=12, grey=0)
	return core.rgvs.RemoveGrain(corrected, mode=[17,17,1])

def dvnrmed(clip, sad=150, scd1=200, scd2=33, blk=8, pel=2, show=False, multiplier=8, threshold=0.05):
	core = vs.get_core()
	scd2=int(255*(scd2/100))

	v1 = clip.fmtc.bitdepth(bits=16)
	inclip = v1
	lowpass = core.resize.Bicubic(v1,width=32,height=32,filter_param_a=1,filter_param_b=0).focus.TemporalSoften(1, 255, 255, 254, 2).resize.Bicubic(width=v1.width,height=v1.height)
	hipass = core.std.Expr(clips=[lowpass,v1], expr=["x y - 32768 +","x y - 32768 +","x y - 32768 +"])
	vsuper = core.mv.Super(hipass,pel=pel)
	vf = core.mv.Analyse(vsuper,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	compf = core.mv.Compensate(hipass,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	vb = core.mv.Analyse(vsuper,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compb = core.mv.Compensate(hipass,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	interleaved = core.std.Interleave([compf,hipass,compb], mismatch=False)
	compf_wiped = core.mv.Compensate(hipass,vsuper,vf,thsad=9999,thscd1=scd1,thscd2=scd2)
	compb_wiped = core.mv.Compensate(hipass,vsuper,vb,thsad=9999,thscd1=scd1,thscd2=scd2)
	interleaved_wiped = core.std.Interleave([compf_wiped,hipass,compb_wiped], mismatch=False)
	wiped = core.rgvs.Clense(interleaved_wiped)
	#degrained = core.focus.TemporalSoften(interleaved, 1, 255, 255, 254, 2)
	degrained = core.misc.AverageFrames(interleaved, [1,1,1])
	mask = core.std.Expr(clips=[wiped, degrained],expr=["x y - abs "+str(threshold)+" 65536 * - "+str(multiplier)+" *","32768","32768"]).std.Maximum().resize.Bicubic(int(v1.width/2),int(v1.height/2),prefer_props=False).std.Maximum().resize.Bicubic(v1.width,v1.height,filter_param_a=1,filter_param_b=0,prefer_props=False)
	v1 = core.std.MaskedMerge(degrained, wiped, mask)
	v1 = core.std.SelectEvery(v1, 3, 1)
	v1 = core.std.Expr(clips=[lowpass,v1], expr=["x y - 32768 +","x y - 32768 +","x y - 32768 +"])
	if show:
		v1 = core.std.StackHorizontal([v1,mask.std.SelectEvery(3,1)])
	return v1

def fpsblur(clip,fpsnum=24000,fpsden=1000, sad=300,scd1=1000,scd2=50,blk=16,pel=2,samples=9,shutter=0.5,exponent=10, stabilize=True, ml=200):
	core = vs.get_core()
	scd2=int(255*(scd2/100))
	ssamples = int(samples/shutter)
	exp=str(exponent)
	
	v1 = clip
	v1 = core.fmtc.bitdepth(v1, bits=32)
	v1 = core.std.Expr(v1, expr=[exp+" x pow 1 - "+exp+" /","",""])
	v1 = core.fmtc.bitdepth(v1, bits=16)

	if stabilize:
		vsuper = core.mv.Super(v1,pel=pel)
		vf = core.mv.Analyse(vsuper,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
		pandata = core.mv.DepanAnalyse(v1, vectors=vf, zoom=True, rot=True)
		v1 = core.mv.DepanStabilise(v1, data=pandata, dxmax=32, dymax=32, zoommax=1.2,rotmax=2,mirror=15,blur=0,method=1)

	blur = core.bilateral.Bilateral(v1, ref=v1, sigmaS=6*v1.height/1080, sigmaR=.025, algorithm=2,planes=[0])
	blur = core.bilateral.Bilateral(blur, ref=blur, sigmaS=16*v1.height/1080, sigmaR=.025, algorithm=2,planes=[1,2])
	vsuper = core.mv.Super(v1,pel=pel)
	vsuperblur = core.mv.Super(blur,pel=pel)
	vf = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=False)
	vb = core.mv.Analyse(vsuperblur,blksize=blk,overlap=int(blk/2),dct=False,isb=True)
	compf = core.mv.Compensate(v1,vsuper,vf,thsad=sad,thscd1=scd1,thscd2=scd2)
	compb = core.mv.Compensate(v1,vsuper,vb,thsad=sad,thscd1=scd1,thscd2=scd2)
	denoise = core.misc.AverageFrames([compf,v1,compb], [1,1,1])

	vsuperdvnr = core.mv.Super(denoise,pel=pel)
	hifps = core.mv.BlockFPS(denoise,vsuperdvnr,vb,vf,fpsnum*ssamples,fpsden,thscd1=scd1,thscd2=scd2,ml=ml)
	hifps = core.misc.AverageFrames(hifps,np.full(samples,1))
	out = core.std.SelectEvery(hifps,ssamples,0)

	v1 = core.fmtc.bitdepth(out, bits=32)
	v1 = core.std.Expr(v1,expr=["x "+exp+" * 1 + log "+exp+" log /","",""])
	v1 = core.fmtc.bitdepth(v1, bits=16)

	return v1

def qfpsblur(clip,fpsnum=24000,fpsden=1000,samples=9,shutter=0.5,exponent=10, stabilize=True, fast=True, denoise=False):
	core = vs.get_core()
	ssamples = int(samples/shutter)
	exp=str(exponent)
	
	v1 = clip
	lofi = core.resize.Lanczos(clip, format=vs.YUV444P8, matrix_s="709", transfer_s="709", primaries_s="709", matrix_in_s="170m", transfer_in_s="601", primaries_in_s="170m")

	v1 = core.fmtc.bitdepth(v1, bits=32)
	v1 = core.std.Expr(v1, expr=[exp+" x pow 1 - "+exp+" /","",""])
	v1 = core.fmtc.bitdepth(v1, bits=16)
	if fast:
		pel=2
		sad=200
	else:
		pel=4
		sad=100

	if stabilize:
		vsuper = core.mv.Super(clip=lofi, pel=1, chroma=True)
		vf = core.mv.Analyse(vsuper,blksize=32,overlap=0,dct=5,isb=False)
		pandata = core.mv.DepanAnalyse(v1, vectors=vf, zoom=True, rot=True)
		v1 = core.mv.DepanStabilise(v1, data=pandata, dxmax=32, dymax=32, zoommax=1.2,rotmax=2,mirror=15,blur=0,method=1)

	vsuperlinlight = core.mv.Super(clip=v1, pel=pel, chroma=True)
	vsuper = core.mv.Super(clip=lofi, pel=pel, chroma=True)
	vf = core.mv.Analyse(super=vsuper, blksize=32, isb=False,dct=5,overlap=0,delta=1, truemotion=True, chroma=False)
	vf = core.mv.Recalculate(vsuper, vf, thsad=sad, blksize=8,overlap=0, dct=7, truemotion=False, chroma=False)
	vb = core.mv.Analyse(super=vsuper, blksize=32, isb=True,dct=5,overlap=0,delta=1, truemotion=True, chroma=False)
	vb = core.mv.Recalculate(vsuper, vb, thsad=sad, blksize=8,overlap=0, dct=7, truemotion=False, chroma=False)
	if denoise:
		compf = core.mv.Compensate(v1,vsuperlinlight,vf,thsad=sad)
		compb = core.mv.Compensate(v1,vsuperlinlight,vb,thsad=sad)
		v1 = core.misc.AverageFrames([compf,v1,compb], [1,1,1])
	msk1 = core.mv.BlockFPS(clip=v1, super=vsuper, mvbw=vb, mvfw=vf, num=fpsnum*ssamples, den=fpsden, blend=False, mode=5,ml=10)
	fps1 = core.mv.FlowFPS(clip=v1, super=vsuperlinlight, mvbw=vb, mvfw=vf, num=fpsnum*ssamples, den=fpsden, blend=False)

	vf3 = core.mv.Analyse(super=vsuper, blksize=32, isb=False,dct=5,overlap=0,delta=3, truemotion=True, chroma=False)
	vf3 = core.mv.Recalculate(vsuper, vf, thsad=sad, blksize=8,overlap=0, dct=7, truemotion=False, chroma=False)
	vb3 = core.mv.Analyse(super=vsuper, blksize=32, isb=True,dct=5,overlap=0,delta=3, truemotion=True, chroma=False)
	vb3 = core.mv.Recalculate(vsuper, vb, thsad=sad, blksize=8,overlap=0, dct=7, truemotion=False, chroma=False)
	msk3 = core.mv.BlockFPS(clip=v1, super=vsuper, mvbw=vb3, mvfw=vf3, num=fpsnum*ssamples, den=fpsden, blend=False, mode=5,ml=10)
	fps3 = core.mv.FlowFPS(clip=v1, super=vsuperlinlight, mvbw=vb, mvfw=vf, num=fpsnum*ssamples, den=fpsden, blend=False)

	expr = "z a < x y ?"
	hifps = core.std.Expr(clips=[fps1, fps3, msk1, msk3], expr=[expr, "",""])
	if fast:
		hifps = fps1
	hifps = core.misc.AverageFrames(hifps,np.full(samples,1))
	out = core.std.SelectEvery(hifps,ssamples,0)

	v1 = core.fmtc.bitdepth(out, bits=32)
	v1 = core.std.Expr(v1,expr=["x "+exp+" * 1 + log "+exp+" log /","",""])
	v1 = core.fmtc.bitdepth(v1, bits=16)

	return v1

#def padtohd(clip):
#	core = vs.get_core()
#	return core.std.AddBorders(clip, left=int((1920-clip.width)/2), right=int(0.5 + (1920-clip.width)/2), top=int((1080-clip.height)/2), bottom=int(0.5 + (1080-clip.height)/2))
#	

def ledfilter(clip,period=59.4,fwidth=9,forder=3,startrow=10,nrows=24, mthreshold=0.002, soft=5):
	core = vs.get_core()
	v1 = core.resize.Lanczos(clip, format=vs.YUV444PS)

	centre=clip.height / period
	f1 = centre-fwidth
	f2 = centre
	f3 = centre+fwidth
	rad=soft
	passes=1
	filtered = core.std.ShufflePlanes(clips=[v1], planes=[0, 1, 2], colorfamily=vs.RGB)
	filtered = core.std.Transpose(filtered).vcfreq.F1Quiver(filter=[4,f1,f3,forder], morph=0, custom=0, test=0, strow=startrow, nrows=24, gamma=0.15).std.Transpose()
	filtered = core.std.ShufflePlanes(clips=[filtered], planes=[0, 1, 2], colorfamily=vs.YUV)
	protectmask = core.std.Expr([v1, filtered], expr=["x y - abs "+str(mthreshold)+" > 0 1 ?","",""]).std.BoxBlur(planes=[1,0],hradius=rad,hpasses=passes,vradius=rad,vpasses=passes).std.ShufflePlanes(planes=[0, 0, 0], colorfamily=vs.YUV)
	v1 = core.std.Expr([v1,filtered,protectmask], expr=["x 1 z - * y z * +","x 1 z - * y z * +","x 1 z - * y z * +"])

	v1 = core.resize.Lanczos(v1, format=vs.YUV444P16)

	return v1