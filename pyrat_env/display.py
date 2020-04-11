import pygame

class Rendering:
    def __init__(self, window_width=1920, window_height=1080):
        self.screen = None
        if not pygame.display.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
