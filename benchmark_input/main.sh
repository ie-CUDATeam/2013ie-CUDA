a=0
n=100000000
while [ $a -ne 10 ]
do
  echo "${a} 回目の処理"
  str="input_N${n}_${a}" 
  ruby create_array.rb ${n} > ./N${n}/${str}
  a=`expr $a + 1`
done
