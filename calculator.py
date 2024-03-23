
print("1.addition")
print("2.subtraction")
print("3.multiplication")
print("4.divide")

while True:
    option=int(input("select ur option:"))
    a=int(input("enter value of a:"))
    b=int(input("enter value of b:"))

# using if else
    if option==1:
        print(a+b)
    elif option==2:
        print(a-b)
    elif option==3:
        print(a*b)
    elif option==4:
        print(a/b)
    else:
        print("invalid option")

    Again= input("do u want to continue?(yes/no):")
    if Again=="no":
        break
    else:
        print("ohh shit! here we go again")
        continue
