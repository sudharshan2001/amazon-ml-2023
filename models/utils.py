from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error

def mape_obj1(preds, dmatrix):
    label = dmatrix.get_label()
    mape_er = mean_absolute_percentage_error(preds, label)
    grad = 1.0-max(0, 100*(1-mape_er))
    return 'mape', grad
