import param

class MySelector(param.Parameterized):
    my_param = param.Selector(objects=['option1', 'option2', 'option3'], default='option1')

    def use_my_param(self):
        if self.my_param == "option1":
            x = 'ver'
        else:
            x = 'hor'
        return x

    print(my_param.owner)

# Example of creating an instance and using the method

selector_instance = MySelector()
selector_instance.use_my_param()
