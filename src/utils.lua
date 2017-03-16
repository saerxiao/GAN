local utils = {}

function utils.dataSampler(mean, std)
  local it = {}
  it.sample = function(N)
    local sample = torch.Tensor():resize(N)
    sample = sample:normal():mul(std):add(mean)  -- :sort()
    return sample
  end
  return it
end

function utils.noiseSampler(range)
  local it = {}
  it.sample = function(N)
--    local sample = torch.linspace(-range, range, N)
--    local random = torch.Tensor(N)
--    random = random:uniform()*0.01
--    return sample + random
    local random = torch.Tensor(N)
    random:uniform(-range, range)
    return random
  end
  return it
end

local function sample(discriminator, generator, dSampler, gSampler, opt)
  local type = discriminator:type()
  -- critic
  local db_x = torch.linspace(-opt.range, opt.range, opt.nSamples)
  db_x = db_x:type(type)
  local db = torch.Tensor(opt.nSamples):type(type)
  for i = 1, math.ceil(opt.nSamples / opt.batchSize) do
    local low = opt.batchSize * (i-1) + 1
    local hi = opt.batchSize * i
    if hi > opt.nSamples then hi = opt.nSamples end
    db[{{low, hi}}] = discriminator:forward(db_x[{{low, hi}}]:view(-1,1))
  end
  
  -- data
  local x = dSampler.sample(opt.nSamples)
  local xpdf = torch.histc(x, opt.nbins, -opt.range, opt.range)
  xpdf = xpdf / xpdf:sum()
  
  -- generated data
  local z = gSampler.sample(opt.nSamples):type(type)
  local gz = z
  if opt.trainGen then
    gz = torch.Tensor(opt.nSamples):type(type)
    for i = 1, math.ceil(opt.nSamples / opt.batchSize) do
      local low = opt.batchSize * (i-1) + 1
      local hi = opt.batchSize * i
      if hi > opt.nSamples then hi = opt. nSamples end
      gz[{{low, hi}}] = generator:forward(z[{{low, hi}}]:view(-1,1))
    end
  end
  local gpdf = torch.histc(gz:float(), opt.nbins, -opt.range, opt.range)
  gpdf = gpdf / gpdf:sum()
  
  return db, xpdf, gpdf
end

function utils.plotDistribution(iter, discriminator, generator, dSampler, gSampler, opt)
  local db, xpdf, gpdf = sample(discriminator, generator, dSampler, gSampler, opt)
  local db_x = torch.linspace(-opt.range, opt.range, opt.nSamples)
  local px = torch.linspace(-opt.range, opt.range, opt.nbins)
  
  local savefile = string.format('plots/gan/iter%d.png', iter)
  gnuplot.pngfigure(savefile)
  gnuplot.title(string.format('iteration %d', iter))
  gnuplot.plot({'discriminator output', db_x, db, '+'}, {'true distribution', px, xpdf, '+'}, {'generated distribution', px, gpdf, '+'})
  gnuplot.plotflush()
end

return utils