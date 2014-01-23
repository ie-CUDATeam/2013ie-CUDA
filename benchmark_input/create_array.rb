
max = 1 << 25
n   = ARGV[0].to_i
puts n
print rand(max)
(n-1).times do |i|
  print " #{rand(max)}"
end
