require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'

local gpuid = 0
local seed = 123
local nSteps = 1000
local print_every = 1
local learningRate = 0.001
local momentum = 0.9
local batchSize = 20
local nData = 1000
local range = 8
local useGenerator = true

local nSamples, nbins = 1000, 100
local type = 'torch.DoubleTensor'
local dataDim = 1
local genHidden = 128

local utils = require 'utils'

if gpuid > -1 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. gpuid .. '...')
    cutorch.setDevice(gpuid + 1)
    cutorch.manualSeed(seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    gpuid = -1 -- overwrite user setting
  end
end

local generator = nn.Sequential()
generator:add(nn.Linear(dataDim, genHidden)):add(nn.ReLU())
generator:add(nn.Linear(genHidden, dataDim))

local discriminator = nn.Sequential()
discriminator:add(nn.Linear(dataDim, genHidden*2)):add(nn.ReLU())
discriminator:add(nn.Linear(genHidden*2, genHidden*2)):add(nn.ReLU())
discriminator:add(nn.Linear(genHidden*2, genHidden*2)):add(nn.ReLU())
discriminator:add(nn.Linear(genHidden*2, 1))
discriminator:add(nn.Sigmoid())

local container = nn.Container()
container:add(generator)
container:add(discriminator)

local criterion = nn.BCECriterion()

-- ship the model to the GPU if desired
if gpuid == 0 then
  container = container:cuda()
  criterion = criterion:cuda()
  type = container:type()
end

local params, grads = container:getParameters()

local Loader = require 'Loader'
local loader = Loader.create(batchSize, nData, dataDim)

local function findHits(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  return torch.eq(predict, target):sum()
end

local function findHitsBi(output, positive)
  if positive then
    return torch.gt(output, 0.5):sum()
  else
    return torch.lt(output, 0.5):sum()
  end
end

local discriminatorAccuracyD,  discriminatorAccuracyG, discriminatorAccuracy = nil, nil, nil
local dataIter = loader:iterator("data")
local noiseIter = loader:iterator("noise")
local dSampler = utils.dataSampler(-4, 0.5)
local gSampler = utils.dataSampler(4, 0.5)
--if useGenerator then
--  gSampler = utils.noiseSampler(range)
--end

local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local gInput = gSampler.sample(batchSize):view(batchSize, dataDim)
  gInput = gInput:type(type)
  local generatedData = gInput
  if useGenerator then
    generatedData = generator:forward(gInput)
  end
  
  local data = dSampler.sample(batchSize):view(batchSize, dataDim)
  data = data:type(type)
  local dInput = torch.cat(data, generatedData, 1)
  
  local dOutput = discriminator:forward(dInput)
  local label = dOutput.new():resize(dOutput:size(1))
  label[{{1, data:size(1)}}]:fill(1)
  label[{{data:size(1)+1, -1}}]:fill(0)
  local hitsD = findHitsBi(dOutput[{{1, data:size(1)}}], true)
  local hitsG = findHitsBi(dOutput[{{data:size(1)+1, -1}}], false)
  discriminatorAccuracyD = hitsD/data:size(1)
  discriminatorAccuracyG = hitsG/data:size(1)
  discriminatorAccuracy = (hitsD + hitsG) / dOutput:size(1)
  
  
  local loss = criterion:forward(dOutput, label)
  local gradLoss = criterion:backward(dOutput, label)
  local gradDInput = discriminator:backward(dInput, gradLoss)
  if useGenerator then
    generator:backward(gInput, -gradDInput[{{data:size(1)+1, -1}}])
  end
  return loss, grads
end

--local function sample()
--  -- decision boundary
--  local db_x = torch.linspace(-range, range, nSamples)
--  db_x = db_x:type(type)
--  local db = torch.Tensor(nSamples):type(type)
--  for i = 1, math.ceil(nSamples / batchSize) do
--    local low = batchSize * (i-1) + 1
--    local hi = batchSize * i
--    if hi > nSamples then hi = nSamples end
--    db[{{low, hi}}] = discriminator:forward(db_x[{{low, hi}}]:view(-1,1))
--  end
--  
--  -- data
--  local x = dSampler.sample(nSamples)
--  local xpdf = torch.histc(x, nbins, -range, range)
--  xpdf = xpdf / xpdf:sum()
--  
--  -- generated data
--  local z = gSampler.sample(nSamples):type(type)
--  local gpdf = nil
--  if useGenerator then
--    local gz = torch.Tensor(nSamples):type(type)
--    for i = 1, math.ceil(nSamples / batchSize) do
--      local low = batchSize * (i-1) + 1
--      local hi = batchSize * i
--      if hi > nSamples then hi = nSamples end
--      gz[{{low, hi}}] = generator:forward(z[{{low, hi}}]:view(-1,1))
--    end
--    gpdf = torch.histc(gz:float(), nbins, -range, range)
--  else
--    gpdf = torch.histc(z:float(), nbins, -range, range)
--  end
--  gpdf = gpdf / gpdf:sum()
--  
--  return db, xpdf, gpdf
--end

--local function plotDist(iter)
--  local db, xpdf, gpdf = sample()
--  local db_x = torch.linspace(-range, range, nSamples)
--  local px = torch.linspace(-range, range, nbins)
--  
--  local savefile = string.format('plots/gan/iter%d.png', iter)
--  gnuplot.pngfigure(savefile)
--  gnuplot.title(string.format('iteration %d', iter))
--  gnuplot.plot({'discriminator output', db_x, db, '+'}, {'true distribution', px, xpdf, '+'}, {'generated distribution', px, gpdf, '+'})
--  gnuplot.plotflush()
--end

local optim_opt = {learningRate = learningRate, momentum = momentum}
for i = 1, nSteps do
  local _, loss = optim.adam(feval, params, optim_opt)
  
  print("i = ", i, " loss = ", loss[1], " discriminatorAccuracy = ", discriminatorAccuracy, " discriminatorAccuracyD = ", discriminatorAccuracyD, " discriminatorAccuracyG = ", discriminatorAccuracyG)
  
  local checkpoint = {}
  checkpoint.iter = i
  checkpoint.generator = generator
  checkpoint.discriminator = discriminator
  local savefile = string.format('saved-model/iter%d.t7', checkpoint.iter)
  
  if i % print_every == 0 then
    utils.plotDistribution(i, discriminator, generator, dSampler, gSampler, opt)
--    plotDist(i)
  end
--  torch.save(savefile, checkpoint)
end