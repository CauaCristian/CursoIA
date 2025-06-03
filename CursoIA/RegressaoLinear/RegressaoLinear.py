from numpy import *

class RegressaoLinear:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__resultadoCorrelacao = self.__correlacao()
        self.__resultadoInclinacao = self.__inclinacao()
        self.__resultadoInterceptacao = self.__interceptacao()

    def __correlacao(self):
        covariacao = cov(self.x, self.y, bias=True)[0][1]
        varianciaX = var(self.x)
        varianciaY = var(self.y)
        return covariacao / sqrt(varianciaX * varianciaY)

    def __inclinacao(self):
        stdX = std(self.x)
        stdY = std(self.y)
        return self.__resultadoCorrelacao * (stdY / stdX)

    def __interceptacao(self):
        mediaX = mean(self.x)
        mediaY = mean(self.y)
        return mediaY - mediaX * self.__resultadoInclinacao

    def previsao(self, valor):
        return self.__resultadoInterceptacao + (self.__resultadoInclinacao * valor)


temperaturaCelsius = [20, 22, 25, 27, 30, 32, 35]
vendasDeSorvete = [100, 115, 150, 160, 180, 195, 220]
modelo = RegressaoLinear(temperaturaCelsius, vendasDeSorvete)
previsao = modelo.previsao(10)
print(previsao)
