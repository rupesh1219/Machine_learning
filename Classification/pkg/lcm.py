# count = 0
# a=True
# b=10
# c=0

# while a==True:
#     for j in range(1,11):
#         if b%j == 0:
#             count += 1
#     if count == 10:
#         print(b)
#     else:
#         b = b+1
#         a=True


def lcm(list_of_numbers):
    maximum = max(list_of_numbers)
    count = 0
    a=True
    while(a==True):
        for i in list_of_numbers:
            if maximum % i == 0:
                count += 1
        if count == len(list_of_numbers):
            return maximum
        else:
            maximum += 1
            a=True
