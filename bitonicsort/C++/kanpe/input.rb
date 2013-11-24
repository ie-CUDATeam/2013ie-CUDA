num = 100
max = 70 
puts num
num.times do |i|
  str = rand(max).to_s
  str += " " if i < num-1
  print str
end
