require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'
require 'MinDiffCriterion'

local utils = require 'utils'

cmd = torch.CmdLine()
-- model options
cmd:option('--genHidden', 64, 'dimention of the hidden layer for the generator')
cmd:option('--dHidden', 128, 'dimention of the hidden layer for the discriminator')

-- data options
cmd:option('-range', 8, 'range for generating the input noise')
cmd:option('-mean', -4, 'mean of the true gaussian distribution')
cmd:option('-std', 0.5, 'std for the true gaussian distribution')
cmd:option('-nSamples', 10000,'number of samples drawn from the distribution to create plots')
cmd:option('-nbins', 100,'number of bins to create the distribution from samples')

-- training options
cmd:option('-batchSize', 64, 'batch size for training in each iteration')
cmd:option('-dIters', 5,'number of iterations to train the descriminator per each generator update')
cmd:option('-gIters', 500,'number of iterations fo generator training')
cmd:option('-learningRate', 0.0005,'training learning rate')
cmd:option('-momentum', 0.9,'momentum for adam')
cmd:option('-clamp', 0.01,'upper bound for weight clamping for the discriminator')
cmd:option('-trainGen', true, 'train generator. False means only to train discriminator')

cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
opt = cmd:parse(arg)

if opt.gpuid > -1 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

local type = 'torch.DoubleTensor'
local dataDim = 1 -- the true distribution is sampled from a gaussian distribution, so its dimention is 1
local dd = nn.Sequential()
dd:add(nn.Linear(dataDim, opt.dHidden)):add(nn.ReLU())
dd:add(nn.Linear(opt.dHidden, opt.dHidden)):add(nn.ReLU())
--dd:add(nn.Linear(dHidden, dHidden)):add(nn.ReLU())
dd:add(nn.Linear(opt.dHidden, 1))
local ddMean = nn.Sequential()
ddMean:add(dd):add(nn.Mean())
local dgMean = ddMean:clone('weight', 'bias', 'gradWeight', 'gradBias')
local discriminator = nn.ParallelTable():add(ddMean):add(dgMean)

local generator = nn.Sequential()
generator:add(nn.Linear(dataDim, opt.genHidden)):add(nn.ReLU())
generator:add(nn.Linear(opt.genHidden, opt.genHidden)):add(nn.ReLU())
--generator:add(nn.Linear(genHidden, genHidden)):add(nn.ReLU())
generator:add(nn.Linear(opt.genHidden, dataDim))

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  discriminator = discriminator:cuda()
  generator = generator:cuda()
  type = generator:type()
end

local paramsD, gradsD = discriminator:getParameters()
--paramsD:uniform(-c, c)
local paramsG, gradsG = generator:getParameters()

local dSampler = utils.dataSampler(opt.mean, opt.std)
--local gSampler = utils.dataSampler(-4, 0.5)
local gSampler = utils.noiseSampler(opt.range)

local function doEvalD(x, z)
  local gz = z
  if opt.trainGen then
    gz = generator:forward(z)
  end
  local output = discriminator:forward({x, gz})
  local loss = output[1] - output[2]
  local one = torch.ones(1):type(type)
  local gradDInput = discriminator:backward({x, gz}, {one, -one})
  return loss, gradDInput
end

local fevalD = function(w)
  if w ~= paramsD then
    paramsD:copy(w)
  end
  gradsD:zero()
  
  local x = dSampler.sample(opt.batchSize):view(-1, 1):type(type)
  local z = gSampler.sample(opt.batchSize):view(-1, 1):type(type)
  local loss, _ = doEvalD(x, z)  
  paramsD:clamp(-opt.clamp, opt.clamp)
  
--  print(gradsD:norm())
  return loss, -gradsD
end

local fevalG = function(w)
  if w ~= paramsG then
    paramsG:copy(w)
  end
  gradsD:zero()
  gradsG:zero()
  
  local x = dSampler.sample(opt.batchSize):view(-1, 1):type(type)
  local z = gSampler.sample(opt.batchSize):view(-1, 1):type(type)
  local loss, gradDInput = doEvalD(x, z)
  generator:backward(z, gradDInput[2])
  return loss, gradsG
end

--local function sample()
--  -- critic
--  local db_x = torch.linspace(-range, range, nSamples)
--  db_x = db_x:type(type)
--  local db = torch.Tensor(nSamples):type(type)
--  for i = 1, math.ceil(nSamples / batchSize) do
--    local low = batchSize * (i-1) + 1
--    local hi = batchSize * i
--    if hi > nSamples then hi = nSamples end
--    db[{{low, hi}}] = dd:forward(db_x[{{low, hi}}]:view(-1,1))
--  end
--  
--  -- data
--  local x = dSampler.sample(nSamples)
--  local xpdf = torch.histc(x, nbins, -range, range)
--  xpdf = xpdf / xpdf:sum()
--  
--  -- generated data
--  local z = gSampler.sample(nSamples):type(type)
--  local gz = z
--  if trainGen then
--    gz = torch.Tensor(nSamples):type(type)
--    for i = 1, math.ceil(nSamples / batchSize) do
--      local low = batchSize * (i-1) + 1
--      local hi = batchSize * i
--      if hi > nSamples then hi = nSamples end
--      gz[{{low, hi}}] = generator:forward(z[{{low, hi}}]:view(-1,1))
--    end
--  end
--  local gpdf = torch.histc(gz:float(), nbins, -range, range)
--  gpdf = gpdf / gpdf:sum()
--  
--  return db, xpdf, gpdf
--end

--local function plotDist(iter)
--  local db, xpdf, gpdf = sample()
--  local db_x = torch.linspace(-range, range, nSamples)
--  local px = torch.linspace(-range, range, nbins)
--  
--  local savefile = string.format('plots/wgan/iter%d.png', iter)
--  gnuplot.pngfigure(savefile)
--  gnuplot.title(string.format('iteration %d', iter))
--  gnuplot.plot({'critic', db_x, db, '+'}, {'data distribution', px, xpdf, '+'}, {'generated distribution', px, gpdf, '+'})
--  gnuplot.plotflush()
--end

local optimOptD = {learningRate = opt.learningRate, momentum = opt.momentum}
local optimOptG = {learningRate = opt.learningRate, momentum = opt.momentum}
for i = 1, opt.gIters do
  local loss = nil
  for j = 1, opt.dIters do
--    optim.adam(fevalD, paramsD, optimOptD)
    _, loss = optim.rmsprop(fevalD, paramsD, optimOptD)
--    print("i = ", i, " loss = ", loss[1][1])
  end
--  local _, loss = optim.adam(fevalG, paramsG, optimOptG)
  if opt.trainGen then
    _, loss = optim.rmsprop(fevalG, paramsG, optimOptG)
  end
--  local _, loss = optim.sgd(feval, params, optim_opt)
  
  print("i = ", i, " loss = ", loss[1][1])
  
--  local checkpoint = {}
--  checkpoint.iter = i
--  checkpoint.generator = generator
--  checkpoint.discriminator = discriminator
--  local savefile = string.format('saved-model/iter%d.t7', checkpoint.iter)
  if i % 10 == 0 then
    utils.plotDistribution(i, discriminator, generator, dSampler, gSampler, opt)
  end
--  torch.save(savefile, checkpoint)
end
