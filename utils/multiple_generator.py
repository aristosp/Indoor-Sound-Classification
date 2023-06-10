
def generator_multiple(generator, df1, df2, batch_size, img_height1, img_width1, img_height2, img_width2, subset, seed, colormode, shuffle):
    x_col1 = 'Filepath'
    x_col2 = 'Filepath'
    y_col1 = 'Label'
    y_col2 = 'Label'
    genX1 = generator.flow_from_dataframe(dataframe=df1, x_col=x_col1, y_col=y_col1,
                                          target_size=(img_height1, img_width1),
                                          class_mode='categorical',
                                          batch_size=batch_size, color_mode=colormode,
                                          shuffle=shuffle, subset=subset,
                                          seed=seed)

    genX2 = generator.flow_from_dataframe(dataframe=df2, x_col=x_col2, y_col=y_col2,
                                          target_size=(img_height2, img_width2),
                                          class_mode='categorical', color_mode=colormode,
                                          batch_size=batch_size,
                                          shuffle=shuffle, subset=subset,
                                          seed=seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label
