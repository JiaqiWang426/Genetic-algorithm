import random
from dataclasses import dataclass, field
import numpy as np



def main(Requirements, Tests, Scores, Durations,
          S_min, w_score, w_duration, M, iteration, l,
          mutation_rate, k):
    #scores 包含了每一个requirement的score
    #Durations包含了每个test的duration
    #M是杂交池的样本
    #每个requirement有一个score，每一个test有一个duration
    #w_score是得分权重
    #w_duration是时长权重
#S_min是得分阈值
#x是向量，w_score和w_time是权重
#M是杂交池的大小
#iteration是迭代次数
#l是等位基因切割点数量
#mutation_rate是变异率
#k是k-tournament参数，见select_population函数
    n = len(Tests)
    m = len(Requirements)
    def random_vector(n):#n是test的数量
        #随机生成np阵列(二进制向量S)
        return np.random.randint(0, 2, size = n, dtype = np.uint8)  

    def duration(x):
        #方案x的时长，越低越好
        return float(np.dot(x, Durations)) #点乘。将每个test的时长加起来

    def score(x):
        #方案x的总得分，越高越好
        pass
    
    def fitness(x) :
        #适应度函数，综合duration和score，越高越好
        S_max = sum(Scores)
        T_max = sum(Durations)
        s_part = (score(x) - S_min) / (S_max - S_min)
        t_part = (duration(x) - 0) /(T_max - 0)
        return w_score * s_part - w_duration * t_part 

    #测试方案x, x包含数个tests。x是numpy阵列
    @dataclass(eq = False)
    class scheme():
        x:np.ndarray
        fitness: float = field(init = False)

        def __post_init__(self):
            self.fitness =fitness(self.x)

        def __hash__(self):
            return hash(self.x.tobytes())
        
        def __eq__(self, other):
            return isinstance(other, scheme) and np.array_equal(self.x)
        
    def select_population(population):
        #k-tournament：每次抽取k个人，选择最优秀（fitness最高的）的。
        # 直到杂交池有M个。
        mating_pool = []
        def select(population):
            #进行一次k-tournament
            winner = max(random.sample(population, k),
                          key = lambda ind: ind.fitness)
            return winner
        selected = set()
        while len(mating_pool) < M:
            winner = select(population)
            selected.add(winner)
            if winner not in selected:
                mating_pool.append(winner)
        return mating_pool

    
    def crossbreed_population(population):
        #杂交
        def crossbreed(father: scheme, mother: scheme):
             #father, mother, self都是长度为n只为0或1的向量
            def mutation(x):
                return
            return 
        return
    
    def replace_population(population, offspring):
        #新一代杂交池
        merged = population + offspring
        merged.sort(key = lambda ind: ind.fitness, reverse = True)
        return merged[:M]
    
    #——以下是主程序——
    population = [random_vector(n) for v in range(M)] 
    select_population(population)
    for i in range(iteration):
        crossbreed_population(population)
        replace_population(population)
    return select_population(population)


'''
'''