import pygame
import math
import numpy as np 
import random
import sys
import os
os.environ["PATH"] += 'C:/Program Files/Graphviz 2.44.1/bin/'

import neat
import pickle


pygame.font.init()  # init font

WIDTH = 900
HEIGHT =500
SCREEN_TITLE = 'Smart Dots'
GRAVITY =9.8



FLOOR = 650
gen=0

STAT_FONT = pygame.font.SysFont("comicsans", 30)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))


cacti_imgs = [pygame.image.load(os.path.join("imgs","Cactus" +str(x) + '.png')) for x  in range(1,4)]
dino_imgs = [ pygame.transform.scale( pygame.image.load(os.path.join("imgs","dino" + str(x) + ".png")),(int(156/3), int(168/3)) ) for x in range(1,4)]
floor_imgs = [pygame.image.load(os.path.join("imgs","Ground" + str(x) + ".png")) for x in range(1,8)]




for i, img in enumerate(cacti_imgs):
	width = img.get_width()
	height = img.get_height()
	cacti_imgs[i] = pygame.transform.scale(cacti_imgs[i], (int(width/3), int(height/3)))


for i, img in enumerate(floor_imgs):
	width = img.get_width()
	height = img.get_height()
	floor_imgs[i] = pygame.transform.scale(floor_imgs[i], (width, int(height/2)))




# ---------------------------------------------------------------------------------------------------------------
class Dino():
	def __init__(self):
		self.y=400
		self.x=200

		self.vel=0
		self.tick_count=0

		self.floorY= HEIGHT-100

		self.IMGS = dino_imgs

		self.img = self.IMGS[1]

		self.tick = 0

	def jump(self):
		self.img = self.IMGS[2]
		if self.vel<-21 or self.vel ==0:
			self.vel = 20.0/2
			self.tick_count=0
			self.height = self.y


	def move(self):

		self.tick_count+=1

		if self.tick_count< 10:
			pass
		else:
			if self.vel>-40 and self.y<=self.floorY:
				self.vel-= (2.75)/2
			elif self.vel>-40:
				pass


		disp = self.vel

	
		if self.y-disp>=self.floorY and disp<0:
			if self.y<self.floorY:
				disp=(self.floorY-self.y )*-1
				self.img = self.IMGS[0]
			else:
				disp=0

		self.y-= disp
		self.y=int(self.y)
 

	def update(self):
		self.tick +=1
		self.move()

		if self.tick % 5==0:
			if self.IMGS.index(self.img) ==0:
				self.img = self.IMGS[1]
			elif self.IMGS.index(self.img) ==1:
				self.img = self.IMGS[0] 

		if self.y>=self.floorY:
			self.vel=0


	def get_mask(self):
		return pygame.mask.from_surface(self.img)




# ---------------------------------------------------------------------------------------------------------------
class Cactus():
	def __init__(self, x, speed):
		self.type=0
		self.x = x
		self.y = 400

		self.speed = speed

		self.alive=True
		self.passed = False

		self.IMGS = cacti_imgs
		self.type = random.randrange(0, len(self.IMGS), 1)
		self.img = self.IMGS[self.type]

	def move(self, speed):
		self.x -= speed

	def update(self, speed):

		if self.x<=-50:
			self.alive = False

		if self.x <= 170:
			self.passed = True

		self.move(speed)

	def draw(self, WIN):
		#pygame.draw.rect(WIN, (0, 150, 0), (self.x, self.y, 40, 80))
		pass

	def collide(self, dino, screen):
		dino_mask = dino.get_mask()
		mask = pygame.mask.from_surface(self.img)

		offset = (self.x - dino.x, self.y-int(dino.y))

		b_point = dino_mask.overlap(mask, offset)

		if b_point:
			return True

		return False



# ---------------------------------------------------------------------------------------------------------------
class Ground():
	def __init__(self, x, speed):
		self.IMGS = floor_imgs
		self.img = self.IMGS[random.randrange(0, 7, 1)]

		self.x = x
		self.y = HEIGHT- self.img.get_height() -50

		self.speed = speed

		self.alive = True


	def move(self, speed):
		self.x -= speed

	def update(self, speed):
		self.move(speed)

		if self.x<=-self.img.get_width()+15:
			self.alive =False



# ---------------------------------------------------------------------------------------------------------------
def draw_window(win, dinos, cacti, next_cactus, grounds, score, gen, speed, dist):
	black= (0, 0, 0)

	win.fill(pygame.Color('white'))

	for ground in grounds:
		win.blit(ground.img, (ground.x, ground.y))
		#pygame.draw.rect(win, (255,0,0), (ground.x, ground.y, 4, 4))

	for cactus in cacti:
		win.blit(cactus.img, (cactus.x, cactus.y))	
		pygame.draw.rect(win, (255, 0,0), (cactus.x, cactus.y, 4, 4))		
		#cactus.draw(win)

	for dino in dinos:
		win.blit(dino.img, (dino.x, int(dino.y)))

		pygame.draw.line(win, (0,0,255), (dino.x, dino. y), (cacti[next_cactus].x, cacti[next_cactus].y) )


	# score
	score_label = STAT_FONT.render("Score: " + str(score),1, black)
	win.blit(score_label, (10, 35))

	# generations
	score_label = STAT_FONT.render("Gens: " + str(gen-1),1, black)
	win.blit(score_label, (10, 10))

	# alive
	score_label = STAT_FONT.render("Alive: " + str(len(dinos)),1, black)
	win.blit(score_label, (10, 60))

	# speed
	score_label = STAT_FONT.render("Speed: " + str(speed),1, black)
	win.blit(score_label, (10, 85))

	# speed
	score_label = STAT_FONT.render("Distance: " + str(dist),1, black)
	win.blit(score_label, (10, 110))


	pygame.display.update()



# ---------------------------------------------------------------------------------------------------------------

def eval_genomes(genomes, config):

	global WIN, gen
	screen = WIN
	gen+=1

	speed = 7

	nets=[]
	dinos=[]
	ge=[]


	for genome_id, genome in genomes:
		genome.fitness =0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		dinos.append(Dino())
		ge.append(genome)


	cacti = []
	next_cactus = 0
	startingX = WIDTH+50
	for i in range(5):
		cacti.append(Cactus(startingX, speed))
		startingX += random.randrange(350, 650, 1)


	grounds = []
	startingX =0
	for i in range(8):
		grounds.append(Ground(startingX, speed))
		startingX += grounds[-1].img.get_width()



	clock = pygame.time.Clock()
	score = 0
	run = True

	dist = 0

	tickSpeed = 120
	while run and len(dinos)>0:
		clock.tick(tickSpeed)

		dist +=1
		if dist %(tickSpeed*5)==0 and speed <11:
			speed+=1

		for event in pygame.event.get():
			if event.type ==pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				break
		

		delete = False
		for i, cactus in enumerate(cacti):
			cactus.update(speed)

			if cactus.alive ==False:
				delete = True
				continue
			if cactus.passed == True:
				next_cactus = i+1

		if delete ==True:
			del cacti[0]
			next_cactus -=1
			score +=1
			cacti.append(Cactus(cacti[-1].x + random.randrange(350, 650, 1), speed))

			for g in ge:
				g.fitness+= 5				# Every new pipe +5 points





		for cactus in cacti:
			for dino in dinos:
				if cactus.collide(dino, screen):
					dino.alive = False
					
					ge[dinos.index(dino)].fitness -= 5				# -5 if death
					nets.pop(dinos.index(dino))
					ge.pop(dinos.index(dino))
					dinos.pop(dinos.index(dino))



		for x, dino in enumerate(dinos):
			#ge[x].fitness = score
			dino.update()

			# Inputs cactus type, dino y, cactus x
			output = nets[dinos.index(dino)].activate((cacti[next_cactus].type , dino.y, cacti[next_cactus].x, speed))

			if output[0]>0.5:
				dino.jump()



		delete = False

		for ground in grounds:
			ground.update(speed)

			if ground.alive ==False:
				delete = True
				continue

		if delete ==True:
			del grounds[0]
			grounds.append(Ground(grounds[-1].x + grounds[-1].img.get_width(), speed))


		
		draw_window(screen, dinos, cacti, next_cactus, grounds, score, gen, speed, dist)


	#node_names = {-1:'Bird Y', -2:'Top Pipe Y', -3:'Bottom Pipe Y', 0:'Output'}
	#draw_net(config, genome, filename='analytics', node_names=node_names)

	#print('Saved Analytics')
	#print(f'Best Score: {score}')
	#sys.exit()
	#plot_stats(neat.StatisticsReporter(), view =True)




# ---------------------------------------------------------------------------------------------------------------

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


    # Create the population, which is the top-level object for a NEAT run.

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 10000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
			





# ---------------------------------------------------------------------------------------------------------------
if __name__ =='__main__':
	local_dir=os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward.txt')
	print(config_path)
	run(config_path)












	'''
	game = Game()
	while True:
		game.update()







class Game():
	def __init__(self):
		global WIN


		self.speed=7

		self.dino = Dino()
		self.screen =WIN

		self.cacti = []

		self.next_cactus = 0

		startingX = WIDTH+50
		for i in range(5):
			self.cacti.append(Cactus(startingX, self.speed))
			startingX += random.randrange(350, 650, 1)


		self.grounds = []
		startingX =0
		for i in range(8):
			self.grounds.append(Ground(startingX, self.speed))
			startingX += self.grounds[-1].img.get_width()



	def manage_cacti(self):
		delete = False
		for i, cactus in enumerate(self.cacti):
			cactus.update()

			if cactus.collide(self.dino, self.screen):
				print('Game Over')

			if cactus.alive ==False:
				delete = True
				continue
			if cactus.passed == True:
				self.next_cactus = i+1

		if delete ==True:
			del self.cacti[0]
			self.next_cactus -=1
			self.cacti.append(Cactus(self.cacti[-1].x + random.randrange(350, 650, 1), self.speed))




	def manage_ground(self):
		delete = False

		for ground in self.grounds:
			ground.update()

			if ground.alive ==False:
				delete = True
				continue

		if delete ==True:
			del self.grounds[0]
			self.grounds.append(Ground(self.grounds[-1].x + self.grounds[-1].img.get_width(), self.speed))


	def update(self):

		clock = pygame.time.Clock()
		while True:
			clock.tick(60)

			self.screen.fill(pygame.Color('white'))

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
						self.dino.jump()
 
			self.manage_cacti()
			self.manage_ground()

			self.dino.update()
			

			draw_window(self.screen, self.dino, self.cacti, self.next_cactus, self.grounds)
'''