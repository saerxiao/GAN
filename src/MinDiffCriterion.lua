local MinDiffCriterion, parent = torch.class('nn.MinDiffCriterion', 'nn.Module')

function MinDiffCriterion:__init()
   parent.__init(self)
end

function MinDiffCriterion:updateOutput(input)
  assert(torch.type(input) == 'table', "input must be a table")
  assert(#input == 2, "input must be a table of two entries")
  self.output = input[1]:mean() - input[2]:mean()
  return self.output
end

function MinDiffCriterion:updateGradInput(input, gradOutput)
  assert(torch.type(input) == 'table', "input must be a table")
  assert(#input == 2, "input must be a table of two entries")
--  local grad1 = (input[2]:mean() - input[1]:mean()) / 2
  local one = input[1].new():resizeAs(input[1])
--  if (input[1]:mean() < input[2]:mean()) then
--    grad1:fill(1)
--  else
--    grad1:fill(-1)
--  end
--  grad1:fill((input[2]:mean() - input[1]:mean()) / 2)
  one:fill(1)
  self.gradInput = {one / input[1]:size(1), -one / input[2]:size(1)}
  return self.gradInput
end