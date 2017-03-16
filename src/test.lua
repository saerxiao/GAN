require 'torch'
require 'gnuplot'
require 'image'
require 'nn'
require 'MinDiffCriterion'
require 'cunn'
require 'cutorch'
--require 'nngraph'

local utils = require 'utils'

local dataDim, dHidden = 1, 3

local dd = nn.Sequential()
dd:add(nn.Linear(dataDim, dHidden)):add(nn.ReLU())
dd:add(nn.Linear(dHidden, 1))
local ddMean = nn.Sequential()
ddMean:add(dd):add(nn.Mean())
local dg = dd:clone('weight', 'bias', 'gradWeight', 'gradBias')
local dgMean = ddMean:clone('weight', 'bias', 'gradWeight', 'gradBias')
local discriminator = nn.ParallelTable():add(ddMean):add(dgMean)

local x = torch.rand(5, 1)
local g = torch.rand(5, 1)

local out = discriminator:forward({x, g})
print(out[1], out[2])
--print(ddMean:forward(x), dgMean:forward(g))
--print(dd:forward(x), dg:forward(g))

discriminator = discriminator:cuda()
local type = discriminator:type()
--dd = dd:cuda()
--dg = dg:cuda()
--local type = dd:type()
x = x:type(type)
g = g:type(type)
local outgpu = discriminator:forward({x, g})
print(outgpu[1], outgpu[2])
--print(ddMean:forward(x), dgMean:forward(g))

--print(dd:forward(x), dg:forward(g))

--local c = nn.MSECriterion()
--x = torch.rand(3)
--print(x)
--label = torch.zeros(3)
--loss = c:forward(x, label)
--print(loss)
--gradInput = c:backward(x, label)
--print(gradInput)

--local dd = nn.Sequential()
--dd:add(nn.Linear(3,4)):add(nn.Tanh()):add(nn.Tanh())
--local dg = dd:clone('weight', 'bias', 'gradWeight', 'gradBias')
--local x = torch.rand(2,3)
--local z = torch.rand(2,3)
--local loss = -(torch.mean(dd:forward(x)) - torch.mean(dg:forward(z)))
--loss:backward()

--local dd = nn.Sequential()
--dd:add(nn.Linear(3, 4)):add(nn.Tanh()):add(nn.Linear(4,1)):add(nn.Sigmoid())
--local dg = dd:clone('weight', 'bias', 'gradWeight', 'gradBias')
--local d = nn.ParallelTable():add(dd):add(dg)
----local d = nn.Sequential()
----d:add(nn.ParallelTable():add(dd):add(dg)):add(nn.CSubTable())
--local params, grads = d:getParameters()
--print(params:size())
--local gradw = dd.modules[1].gradWeight
--local gradwdg = dg.modules[1].gradWeight
----print(gradw[1], gradwdg[1])
--gradw[1] = 0.53
----print(gradw[1], gradwdg[1])
--local x = torch.rand(2,3)
--local z = torch.rand(2,3)
--local out = d:forward({x, z})
--print(out[1], out[2])
--
--local criterion = nn.MinDiffCriterion()
--local loss = criterion:forward(out)
--print(loss)
--local dloss = criterion:backward(out)
--local gradInput = d:backward({x, z}, dloss)
--print(gradInput[1], gradInput[2])

--local criterion = nn.CSubTable()
--local label = torch.zeros(1)
--local loss = criterion:forward(out, label)
--print(loss)
--local gradLoss = criterion:backward(out, label)
--print(gradLoss[1], gradLoss[2])
--local criterion = nn.ParallelCriterion():add(nn.Itendity()):add(nn.Mean(), -1)
--local loss = criterion:forward(out, {})
--print(loss)
--local gradLoss = criterion:backward(out, {})
--print(gradLoss)

--local dSampler = utils.dataSampler(4, 0.5)
--local range = 8
--local nSamples, nbins = 10000, 100
--local x = dSampler.sample(nSamples)
--local xpdf = torch.histc(x, nbins, -range, range)
--xpdf = xpdf / xpdf:sum()
--
--local nSampler = utils.noiseSampler(range)
--local z = nSampler.sample(nSamples)
--local zpdf = torch.histc(z, nbins, -range, range)
--zpdf = zpdf / zpdf:sum()
--  
--local px = torch.linspace(-range, range, nbins)
--
--gnuplot.pngfigure('plots/test.png')
--gnuplot.plot({px, zpdf, '+'}, {px, xpdf, '+'})
--gnuplot.ylabel('probability')
--gnuplot.plotflush()

--local batchSize = 20
--local nData = 1000
--local dataDim = 50
--
--local Loader = require 'Loader'
--local loader = Loader.create(batchSize, nData, dataDim)
--
--local dataIter = loader:iterator("data")
--local noiseIter = loader:iterator("noise")
--
--local x = torch.Tensor(dataDim)
--for i = 1, dataDim do
--  x[i] = i
--end
--
--local data = dataIter.nextBatch()
--local noise = noiseIter.nextBatch()
--
--gnuplot.raw('set multiplot layout 1, 2')
--gnuplot.plot(x, data[2], '+')
--gnuplot.plot(x, noise[2], '+')