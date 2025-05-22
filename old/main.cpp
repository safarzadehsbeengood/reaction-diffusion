#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <random>
#include <vector>

// Define simulation parameters
const int WIDTH = 800;
const int HEIGHT = 800;
const float DIFFUSION_RATE_A = 1.0f;
const float DIFFUSION_RATE_B = 0.5f;
float FEED_RATE = 0.055f;
float KILL_RATE = 0.062f;

// Structure to represent a cell
struct Cell {
  float a;
  float b;
};

// Function to initialize the grid with random values
std::vector<std::vector<Cell>> initializeGrid() {
  std::vector<std::vector<Cell>> grid(HEIGHT, std::vector<Cell>(WIDTH));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0f, 1.0f);

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      grid[y][x].a = 1.0f;
      grid[y][x].b = 0.0f;

      // Add a small "seed" of B in the center for interesting patterns
      if (x > WIDTH / 2 - 20 && x < WIDTH / 2 + 20 && y > HEIGHT / 2 - 20 &&
          y < HEIGHT / 2 + 20) {
        grid[y][x].b = 1.0f;
      } else {
        grid[y][x].b = dist(gen);
      }
    }
  }
  return grid;
}

float laplaceA(int y, int x, std::vector<std::vector<Cell>> &grid) {
  float sum = 0;
  sum += grid[x][y].a * -1;
  sum += grid[x - 1][y].a * 0.2;
  sum += grid[x + 1][y].a * 0.2;
  sum += grid[x][y + 1].a * 0.2;
  sum += grid[x][y - 1].a * 0.2;
  sum += grid[x - 1][y - 1].a * 0.05;
  sum += grid[x + 1][y - 1].a * 0.05;
  sum += grid[x + 1][y + 1].a * 0.05;
  sum += grid[x - 1][y + 1].a * 0.05;
  return sum;
}

float laplaceB(int y, int x, std::vector<std::vector<Cell>> &grid) {
  float sum = 0;
  sum += grid[x][y].b * -1;
  sum += grid[x - 1][y].b * 0.2;
  sum += grid[x + 1][y].b * 0.2;
  sum += grid[x][y + 1].b * 0.2;
  sum += grid[x][y - 1].b * 0.2;
  sum += grid[x - 1][y - 1].b * 0.05;
  sum += grid[x + 1][y - 1].b * 0.05;
  sum += grid[x + 1][y + 1].b * 0.05;
  sum += grid[x - 1][y + 1].b * 0.05;
  return sum;
}

void updateGrid(std::vector<std::vector<Cell>> &grid) {
  std::vector<std::vector<Cell>> nextGrid = grid;

  // gray-scott model
  for (int y = 1; y < HEIGHT - 1; ++y) {
    for (int x = 1; x < WIDTH - 1; ++x) {
      float a = grid[y][x].a;
      float b = grid[y][x].b;

      float laplacianA = laplaceA(x, y, grid);
      float laplacianB = laplaceB(x, y, grid);

      nextGrid[y][x].a = a + (DIFFUSION_RATE_A * laplacianA) - (a * b * b) +
                         (FEED_RATE * (1 - a));
      nextGrid[y][x].b = b + (DIFFUSION_RATE_B * laplacianB) + (a * b * b) -
                         ((KILL_RATE + FEED_RATE) * b);

      nextGrid[y][x].a = std::max(0.0f, std::min(1.0f, nextGrid[y][x].a));
      nextGrid[y][x].b = std::max(0.0f, std::min(1.0f, nextGrid[y][x].b));
    }
  }
  grid = nextGrid;
}

int main() {
  std::vector<std::vector<Cell>> grid = initializeGrid();
  std::cout << "WIDTH: " << WIDTH << " HEIGHT: " << HEIGHT << std::endl;
  sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Diffusion");
  window.setFramerateLimit(60);
  sf::Image image({WIDTH, HEIGHT}, sf::Color::Black);
  sf::Texture texture;
  sf::Font font("/System/Library/Fonts/NewYork.ttf");

  while (window.isOpen()) {
    while (const std::optional event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>()) {
        window.close();
      } else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->scancode) {
          case sf::Keyboard::Scan::Up:
            KILL_RATE += 0.001;
            break;
          case sf::Keyboard::Scan::Down:
            KILL_RATE -= 0.001;
            break;
          case sf::Keyboard::Scan::Left:
            FEED_RATE -= 0.001;
            break;
          case sf::Keyboard::Scan::Right:
            FEED_RATE += 0.001;
            break;
          default:
            break;
        }
      } else if (const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()) {
        sf::Vector2i mousePos = mouseMoved->position;
        int x = mousePos.x;
        int y = mousePos.y;
        grid[y][x].b = 1.0f;
        int stroke = 10;
        for (int i = y-stroke/2; i < y+stroke/2; i++) {
          if (i < 0 || i >= HEIGHT) continue;
          for (int j = x-stroke/2; j < x+stroke/2; j++) {
            if (j < 0 || j >= WIDTH) continue;
            grid[i][j].b = 1.0f;
            grid[i][j].a = 0.0f;
          }
        }
      }
    }
    window.setActive();
    updateGrid(grid);
    // std::cout << "update" << std::endl;

    // Update the image with the new grid data
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        Cell c = grid[y][x];
        uint8_t value = static_cast<uint8_t>((c.a-c.b)*255);
        image.setPixel(sf::Vector2u(x, y), sf::Color(value, value, value));
      }
    }
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    window.clear();

    sf::String kr(std::string("kr: ").append(std::to_string(KILL_RATE)));
    sf::Text killRate(font, kr);
    killRate.setFillColor(sf::Color::Black);
    sf::String fr(std::string("fr: ").append(std::to_string(FEED_RATE)));
    sf::Text feedRate(font, fr);
    feedRate.setFillColor(sf::Color::Black);
    feedRate.setPosition(sf::Vector2f(0.0, 30.0));

    window.draw(sprite);
    window.draw(killRate);
    window.draw(feedRate);
    window.display();
  }

  return 0;
}
