require 'torch'

local Loader = {}
Loader.__index = Loader

function Loader.create(batchSize, nData, dim)
  local self = {}
  setmetatable(self, Loader)
  local mean = 4
  local std = 1
  local range = 10
  
  self.nBatches = math.ceil( nData/batchSize)
  local nSamples = batchSize * self.nBatches
  local dataSamples = torch.Tensor(nSamples, dim)
  local noiseSamples = torch.Tensor(nSamples, dim)
  for i = 1, nSamples do 
    local dataSample = torch.Tensor():resize(dim)
    dataSamples[i] = dataSample:normal():mul(std):add(mean):sort()
    local noiseSample = torch.Tensor():resize(dim)
    noiseSamples[i] = noiseSample:uniform(-range, range)
  end
  self.data = dataSamples:split(batchSize)
  self.noise = noiseSamples:split(batchSize)
  
  return self
end

function Loader:iterator(type)
  local it = {}
  local cursor = 0
  it.reset = function()
    cursor = 0
  end
  it.nextBatch = function()
    cursor = cursor + 1
    if cursor > self.nBatches then
      cursor = 1
    end
    local batch = nil
    if type == "data" then
      batch = self.data[cursor]
    elseif type == "noise" then
      batch = self.noise[cursor]
    else
      print("unknown type ", type)
      error()
    end
    
    return batch
  end
  return it
end

return Loader