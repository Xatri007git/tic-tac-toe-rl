import pygame
import sys
import pickle
import numpy as np
from math import inf
from game import Game
from player import RLPlayer

# --- Configuration & Palette ---
GAME_BOARD_SIZE = 540
PANEL_WIDTH = 400   # Width of the side panel
MARGIN = 30
HEADER_SIZE = 110
FOOTER_SIZE = 140

# Window Dimensions
WINDOW_WIDTH = GAME_BOARD_SIZE + PANEL_WIDTH + (MARGIN * 3)
WINDOW_HEIGHT = HEADER_SIZE + GAME_BOARD_SIZE + FOOTER_SIZE

# Grid Sizes
GRID_SIZE = GAME_BOARD_SIZE // 3
HEATMAP_SIZE = 300  # Smaller board for the panel
HEATMAP_GRID = HEATMAP_SIZE // 3

LINE_WIDTH = 8
SYMBOL_WIDTH = 12
Q_TABLE_FILE = "q_table_sarsa_ofpol_v5.pkl"

# --- DARK MODE PALETTE ---
BG_COLOR = (30, 30, 35)          
PANEL_BG = (25, 25, 30)          # Darker background for side panel
HEADER_BG = (40, 40, 45)         
GRID_COLOR = (80, 80, 90)        

# Game Pieces 
P1_COLOR = (0, 255, 255)         # Cyan (Human/Minimax)
P2_COLOR = (255, 50, 80)         # Neon Red (RL Agent)

# Text
TEXT_PRIMARY = (255, 255, 255)   
TEXT_SECONDARY = (180, 180, 180) 
LABEL_COLOR = (255, 215, 0)      

# Buttons 
BTN_PRIMARY = (0, 200, 83)       
BTN_SECONDARY = (255, 140, 0)    
BTN_TOGGLE_ON = (255, 60, 60)    
BTN_TOGGLE_OFF = (100, 100, 110) 
BTN_TEXT = (255, 255, 255)

# --- CORRECTED MINIMAX CLASS ---
class MinimaxPlayer:
    def __init__(self): pass

    def get_move(self, game, player):
        # We start the recursion. 'player' is the agent (1 or -1).
        result = self.minimax(game, 9, player, -inf, inf)
        return result[0], result[1] 

    def minimax(self, game, depth, maximize_player, alpha, beta):
        # maximize_player: The ID of the Minimax Agent (does not change during recursion)
        status = game.game_status()
        
        # Base Case: Game Over or Depth Limit
        if depth == 0 or status is not None:
            if status == maximize_player:
                return [-1, -1, 10 + depth] # Win (prefer faster wins)
            elif status == -maximize_player:
                return [-1, -1, -10 - depth] # Loss (prefer slower losses)
            elif status == 0:
                return [-1, -1, 0] # Draw
            return [-1, -1, 0]

        legal_moves = game.legal_moves()
        
        # Determine if it is currently the maximizing player's turn
        is_maximizing = (game.current_player == maximize_player)
        
        if is_maximizing:
            best_score = -inf
            best_move = [-1, -1, -inf]
            
            for move in legal_moves:
                row, col = move // 3, move % 3
                game.make_move(row, col)
                score = self.minimax(game, depth - 1, maximize_player, alpha, beta)
                game.unmake_move(row, col)
                
                # Update Best
                if score[2] > best_score:
                    best_score = score[2]
                    best_move = [row, col, best_score]
                
                # Alpha-Beta Pruning
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_move
            
        else: # Minimizing Player (The Opponent)
            best_score = inf
            best_move = [-1, -1, inf]
            
            for move in legal_moves:
                row, col = move // 3, move % 3
                game.make_move(row, col)
                score = self.minimax(game, depth - 1, maximize_player, alpha, beta)
                game.unmake_move(row, col)
                
                # Update Best (Minimize)
                if score[2] < best_score:
                    best_score = score[2]
                    best_move = [row, col, best_score]
                
                # Alpha-Beta Pruning
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_move

# --- GUI Class ---
class TicTacToeGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe AI Dashboard")
        
        # Game Objects
        self.game = Game()
        self.rl_agent = RLPlayer()
        self.minimax_agent = MinimaxPlayer()
        self.load_q_table()
        
        # State
        self.running = True
        self.game_over = False
        self.is_human_mode = True  
        self.opponent_turn = True 
        self.ai_is_player_1 = False 
        self.end_message = ""
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.btn_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.status_font = pygame.font.SysFont("Arial", 22)
        self.msg_font = pygame.font.SysFont("Arial", 50, bold=True)
        self.label_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.qval_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.panel_title_font = pygame.font.SysFont("Arial", 20, bold=True)
        
        # --- BUTTON LAYOUTS ---
        # 1. Agent Order (Top Center of Game Board)
        self.order_btn_rect = pygame.Rect(0, 0, 180, 40)
        center_x = MARGIN + GAME_BOARD_SIZE // 2
        self.order_btn_rect.center = (center_x, HEADER_SIZE // 2)
        
        # 2. Footer Buttons (Aligned to Game Board)
        self.mode_btn_rect = pygame.Rect(MARGIN, WINDOW_HEIGHT - 80, 180, 50)
        self.restart_btn_rect = pygame.Rect(MARGIN + GAME_BOARD_SIZE - 120, WINDOW_HEIGHT - 80, 120, 50)

    def load_q_table(self):
        try:
            with open(Q_TABLE_FILE, "rb") as f:
                self.rl_agent.qtable = pickle.load(f)
            print(f"Loaded {len(self.rl_agent.qtable)} states.")
        except FileNotFoundError:
            self.rl_agent.qtable = {}

    def draw_button(self, rect, color, text, hover=False):
        shadow_rect = rect.copy()
        shadow_rect.y += 4
        pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=12)
        
        draw_color = tuple(min(c + 30, 255) for c in color) if hover else color
        pygame.draw.rect(self.screen, draw_color, rect, border_radius=12)
        
        txt_surf = self.btn_font.render(text, True, BTN_TEXT)
        txt_rect = txt_surf.get_rect(center=rect.center)
        self.screen.blit(txt_surf, txt_rect)

    def get_color_for_q(self, val, min_v, max_v):
        """Returns a color from Red (Bad) to Green (Good)"""
        if max_v == min_v: return (100, 100, 100) # Neutral Grey
        
        # Normalize 0 to 1
        ratio = (val - min_v) / (max_v - min_v)
        
        # Interpolate Red to Green
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        b = 50
        return (r, g, b)

    def draw_panel(self):
        """Draws the side panel with Heatmap"""
        panel_x = MARGIN + GAME_BOARD_SIZE + MARGIN
        
        # Background
        bg_rect = pygame.Rect(panel_x, HEADER_SIZE, PANEL_WIDTH, GAME_BOARD_SIZE)
        pygame.draw.rect(self.screen, PANEL_BG, bg_rect, border_radius=15)
        
        # Title
        title = self.panel_title_font.render("AGENT MEMORY (Q-Values)", True, TEXT_SECONDARY)
        self.screen.blit(title, (panel_x + 20, HEADER_SIZE + 20))
        
        # --- DRAW HEATMAP GRID ---
        heatmap_x = panel_x + (PANEL_WIDTH - HEATMAP_SIZE) // 2
        heatmap_y = HEADER_SIZE + 60
        
        # Get Q-Values
        state = self.game.game_state()
        actions = self.rl_agent.qtable.get(state, {})
        legal_moves = self.game.legal_moves()
        
        # Calculate Min/Max for color scaling
        relevant_qs = [actions[m] for m in legal_moves if m in actions]
        min_q = min(relevant_qs) if relevant_qs else -1
        max_q = max(relevant_qs) if relevant_qs else 1

        # Draw Cells
        for i in range(9):
            row = i // 3
            col = i % 3
            
            x = heatmap_x + col * HEATMAP_GRID
            y = heatmap_y + row * HEATMAP_GRID
            
            rect = pygame.Rect(x, y, HEATMAP_GRID - 4, HEATMAP_GRID - 4)
            
            # Determine Color
            if i in actions and i in legal_moves:
                val = actions[i]
                color = self.get_color_for_q(val, min_q, max_q)
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                # Draw Value Text
                val_str = f"{val:.2f}"
                txt = self.qval_font.render(val_str, True, (255, 255, 255))
                txt_rect = txt.get_rect(center=rect.center)
                self.screen.blit(txt, txt_rect)
            else:
                # Illegal move or empty
                pygame.draw.rect(self.screen, (50, 50, 60), rect, border_radius=5)
                
                # If occupied on real board, show symbol faintly
                try:
                    # Quick hack to get board val
                    s = self.game.game_state()
                    s = s.replace('[', '').replace(']', '').split()
                    board_val = int(s[i])
                    if board_val != 0:
                        sym = "X" if board_val == 1 else "O"
                        stxt = self.qval_font.render(sym, True, (80, 80, 90))
                        self.screen.blit(stxt, stxt.get_rect(center=rect.center))
                except: pass

        # Draw Legend
        legend_y = heatmap_y + HEATMAP_SIZE + 20
        start_txt = self.btn_font.render("Bad (-1.0)", True, (255, 50, 50))
        end_txt = self.btn_font.render("Good (+1.0)", True, (50, 255, 50))
        self.screen.blit(start_txt, (heatmap_x, legend_y))
        self.screen.blit(end_txt, (heatmap_x + HEATMAP_SIZE - 80, legend_y))

    def draw_ui(self):
        self.screen.fill(BG_COLOR)
        
        # HEADER
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, WINDOW_WIDTH, HEADER_SIZE))
        pygame.draw.line(self.screen, (60, 60, 70), (0, HEADER_SIZE), (WINDOW_WIDTH, HEADER_SIZE), 2)
        
        # Main Title
        title = self.title_font.render("Tic-Tac-Toe AI", True, TEXT_PRIMARY)
        self.screen.blit(title, (MARGIN, 20))
        
        mouse_pos = pygame.mouse.get_pos()
        
        # Order Toggle
        toggle_col = BTN_TOGGLE_ON if self.ai_is_player_1 else BTN_TOGGLE_OFF
        toggle_txt = "Agent: 1ST" if self.ai_is_player_1 else "Agent: 2ND"
        self.draw_button(self.order_btn_rect, toggle_col, toggle_txt, self.order_btn_rect.collidepoint(mouse_pos))

        # --- DRAW GAME BOARD GRID ---
        start_y = HEADER_SIZE
        for i in range(1, 3):
            x = MARGIN + i * GRID_SIZE
            pygame.draw.line(self.screen, GRID_COLOR, (x, start_y), (x, start_y + GAME_BOARD_SIZE), LINE_WIDTH)
            y = start_y + i * GRID_SIZE
            pygame.draw.line(self.screen, GRID_COLOR, (MARGIN, y), (MARGIN + GAME_BOARD_SIZE, y), LINE_WIDTH)
        
        # --- DRAW SIDE PANEL ---
        self.draw_panel()

        # FOOTER
        mode_str = "HUMAN" if self.is_human_mode else "MINIMAX"
        opp_surf = self.label_font.render(f"OPPONENT: {mode_str}", True, LABEL_COLOR)
        self.screen.blit(opp_surf, (MARGIN, WINDOW_HEIGHT - 135))

        if self.game_over:
            stat_txt = "Game Finished"
            stat_col = TEXT_SECONDARY
        else:
            if self.opponent_turn:
                stat_txt = "Your Turn" if self.is_human_mode else "Minimax Calculation..."
                stat_col = P1_COLOR
            else:
                stat_txt = "RL Agent Thinking..."
                stat_col = P2_COLOR
        
        stat_surf = self.status_font.render(stat_txt, True, stat_col)
        # Position status relative to game board width
        stat_rect = stat_surf.get_rect(topright=(MARGIN + GAME_BOARD_SIZE, WINDOW_HEIGHT - 135))
        self.screen.blit(stat_surf, stat_rect)
        
        mode_label = "Switch to AI" if self.is_human_mode else "Switch to Human"
        self.draw_button(self.mode_btn_rect, BTN_PRIMARY, mode_label, self.mode_btn_rect.collidepoint(mouse_pos))
        self.draw_button(self.restart_btn_rect, BTN_SECONDARY, "Restart", self.restart_btn_rect.collidepoint(mouse_pos))

    def draw_marks(self):
        try:
            board = self.game.board 
        except AttributeError:
            s = self.game.game_state()
            s = s.replace('[', '').replace(']', '').split()
            board = [int(x) for x in s]
        if hasattr(board, 'flatten'): board = board.flatten()

        for i in range(9):
            row = i // 3
            col = i % 3
            val = board[i]
            
            center_x = MARGIN + col * GRID_SIZE + GRID_SIZE // 2
            center_y = HEADER_SIZE + row * GRID_SIZE + GRID_SIZE // 2

            if val == 1: 
                s = 40
                pygame.draw.line(self.screen, (0, 100, 100), (center_x-s, center_y-s), (center_x+s, center_y+s), SYMBOL_WIDTH+4)
                pygame.draw.line(self.screen, (0, 100, 100), (center_x-s, center_y+s), (center_x+s, center_y-s), SYMBOL_WIDTH+4)
                pygame.draw.line(self.screen, P1_COLOR, (center_x-s, center_y-s), (center_x+s, center_y+s), SYMBOL_WIDTH)
                pygame.draw.line(self.screen, P1_COLOR, (center_x-s, center_y+s), (center_x+s, center_y-s), SYMBOL_WIDTH)
            elif val == -1: 
                pygame.draw.circle(self.screen, (100, 0, 20), (center_x, center_y), 50, SYMBOL_WIDTH+4) 
                pygame.draw.circle(self.screen, P2_COLOR, (center_x, center_y), 50, SYMBOL_WIDTH)
                pygame.draw.circle(self.screen, BG_COLOR, (center_x, center_y), 50 - SYMBOL_WIDTH) 

    def draw_game_over_msg(self):
        if not self.game_over or not self.end_message: return
        
        # Center the card over the GAME BOARD, not the whole window
        center_x = MARGIN + GAME_BOARD_SIZE // 2
        center_y = HEADER_SIZE + GAME_BOARD_SIZE // 2
        
        card_w, card_h = 420, 150
        card_rect = pygame.Rect(0, 0, card_w, card_h)
        card_rect.center = (center_x, center_y)
        
        card_surface = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surface.fill((255, 255, 255, 180)) 
        self.screen.blit(card_surface, card_rect.topleft)
        
        pygame.draw.rect(self.screen, P1_COLOR, card_rect, width=3)
        
        txt = self.msg_font.render(self.end_message, True, (40, 40, 40))
        txt_rect = txt.get_rect(center=card_rect.center)
        self.screen.blit(txt, txt_rect)
        
        sub = self.status_font.render("Press Restart to play again", True, (80, 80, 80))
        sub_rect = sub.get_rect(center=(center_x, card_rect.bottom + 30))
        self.screen.blit(sub, sub_rect)

    def get_rl_move(self):
        legal = self.game.legal_moves()
        state = self.game.game_state()
        if state in self.rl_agent.qtable:
            acts = self.rl_agent.qtable[state]
            best = legal[0]
            for m in legal:
                if acts[m] > acts[best]: best = m
            return best
        return np.random.choice(legal)

    def process_turn(self, row, col):
        if row is None: return 
        self.game.make_move(row, col)
        self.opponent_turn = not self.opponent_turn
        
        self.draw_ui()
        self.draw_marks()
        pygame.display.update()

        stat = self.game.game_status()
        if stat is not None:
            self.game_over = True
            if stat == 0: self.end_message = "Draw!"
            elif stat == 1: self.end_message = "Opponent Wins!"
            else: self.end_message = "RL Agent Wins!"

    def handle_click(self, pos):
        x, y = pos

        # Buttons
        if self.order_btn_rect.collidepoint(pos):
            self.reset_game(not self.ai_is_player_1)
            return
        if self.restart_btn_rect.collidepoint(pos):
            self.reset_game(self.ai_is_player_1)
            return
        if self.mode_btn_rect.collidepoint(pos):
            self.is_human_mode = not self.is_human_mode
            self.reset_game(self.ai_is_player_1)
            return

        # Board Click (Restricted to Game Board Area)
        if not self.game_over and self.is_human_mode and self.opponent_turn:
            if HEADER_SIZE < y < HEADER_SIZE + GAME_BOARD_SIZE:
                if MARGIN < x < MARGIN + GAME_BOARD_SIZE:
                    adj_x = x - MARGIN
                    adj_y = y - HEADER_SIZE
                    c, r = adj_x // GRID_SIZE, adj_y // GRID_SIZE
                    idx = r * 3 + c
                    if idx in self.game.legal_moves():
                        self.process_turn(r, c)

    def reset_game(self, ai_starts):
        self.game = Game()
        self.game_over = False
        self.end_message = ""
        self.ai_is_player_1 = ai_starts
        self.opponent_turn = not ai_starts

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN: self.handle_click(pygame.mouse.get_pos())
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h: self.reset_game(False)
                    if event.key == pygame.K_a: self.reset_game(True)

            if not self.game_over:
                if not self.opponent_turn: 
                    pygame.time.wait(400) 
                    move = self.get_rl_move()
                    self.process_turn(move // 3, move % 3)
                elif self.opponent_turn and not self.is_human_mode: 
                    pygame.event.pump() 
                    r, c = self.minimax_agent.get_move(self.game, 1) 
                    self.process_turn(r, c)

            self.draw_ui()
            self.draw_marks()
            if self.game_over: self.draw_game_over_msg()
            pygame.display.flip()


gui = TicTacToeGUI()
gui.run()