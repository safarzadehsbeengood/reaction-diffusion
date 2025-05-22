#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <random>
#include <vector>

// Define simulation parameters
const int WIDTH = 1920;
const int HEIGHT = 1080;
const double DIFFUSION_RATE_A = 1;
const double DIFFUSION_RATE_B = 0.5;
double FEED_RATE = 0.054;
double KILL_RATE = 0.062;

// Structure to represent a cell
struct Cell {
  double a;
  double b;
};

int get_idx_from_xy(int x, int y) {
  return y * WIDTH + x;
}

// Function to initialize the arr with random values
std::vector<Cell> initializearr() {
  std::vector<Cell> arr(WIDTH*HEIGHT);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int idx = y * WIDTH + x;
      arr[idx].a = 1.0;
      arr[idx].b = 0.0;

      // Add a small "seed" of B in the center for interesting patterns
      if (x > WIDTH / 2 - 20 && x < WIDTH / 2 + 20 && y > HEIGHT / 2 - 20 &&
          y < HEIGHT / 2 + 20) {
        arr[idx].b = 1.0;
      } else {
        arr[idx].b = dist(gen);
      }
    }
  }
  return arr;
}

double laplaceA(int x, int y, std::vector<Cell> &arr) {
  double sum = 0;
  sum += arr[get_idx_from_xy(x, y)].a * -1;
  sum += arr[get_idx_from_xy(x - 1, y)].a * 0.2;
  sum += arr[get_idx_from_xy(x + 1, y)].a * 0.2;
  sum += arr[get_idx_from_xy(x, y + 1)].a * 0.2;
  sum += arr[get_idx_from_xy(x, y - 1)].a * 0.2;
  sum += arr[get_idx_from_xy(x - 1, y - 1)].a * 0.05;
  sum += arr[get_idx_from_xy(x + 1, y - 1)].a * 0.05;
  sum += arr[get_idx_from_xy(x + 1, y + 1)].a * 0.05;
  sum += arr[get_idx_from_xy(x - 1, y + 1)].a * 0.05;
  return sum;
}


double laplaceB(int x, int y, std::vector<Cell> &arr) {
  double sum = 0;
  sum += arr[get_idx_from_xy(x, y)].b * -1;
  sum += arr[get_idx_from_xy(x - 1, y)].b * 0.2;
  sum += arr[get_idx_from_xy(x + 1, y)].b * 0.2;
  sum += arr[get_idx_from_xy(x, y + 1)].b * 0.2;
  sum += arr[get_idx_from_xy(x, y - 1)].b * 0.2;
  sum += arr[get_idx_from_xy(x - 1, y - 1)].b * 0.05;
  sum += arr[get_idx_from_xy(x + 1, y - 1)].b * 0.05;
  sum += arr[get_idx_from_xy(x + 1, y + 1)].b * 0.05;
  sum += arr[get_idx_from_xy(x - 1, y + 1)].b * 0.05;
  return sum;
}

void updatearr(std::vector<Cell> &arr, std::vector<Cell> &nextarr) {
  // gray-scott model
  for (int y = 1; y < HEIGHT - 1; ++y) {
    for (int x = 1; x < WIDTH - 1; ++x) {
      int idx = get_idx_from_xy(x, y);
      double a = arr[idx].a;
      double b = arr[idx].b;

      double laplacianA = laplaceA(x, y, arr);
      double laplacianB = laplaceB(x, y, arr);

      nextarr[idx].a = a + (DIFFUSION_RATE_A * laplacianA) - (a * b * b) +
                         (FEED_RATE * (1 - a));
      nextarr[idx].b = b + (DIFFUSION_RATE_B * laplacianB) + (a * b * b) -
                         ((KILL_RATE + FEED_RATE) * b);

      nextarr[idx].a = std::max(0.0, std::min(1.0, nextarr[idx].a));
      nextarr[idx].b = std::max(0.0, std::min(1.0, nextarr[idx].b));
    }
  }
  arr.swap(nextarr);
}

int main() {
  std::vector<Cell> arr = initializearr();
  std::vector<Cell> nextarr = arr;
  std::cout << "WIDTH: " << WIDTH << " HEIGHT: " << HEIGHT << std::endl;
  sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Diffusion");
  window.setFramerateLimit(60);
  sf::Image image({WIDTH, HEIGHT}, sf::Color::Black);
  sf::Texture texture;
  sf::Font font("/System/Library/Fonts/NewYork.ttf");

  std::cout << "KILL: " << KILL_RATE << std::endl;
  std::cout << "FEED: " << FEED_RATE << std::endl;

  while (window.isOpen()) {
    while (const std::optional event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>()) {
        window.close();
      } 
    else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
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
        std::cout << "KILL: " << KILL_RATE << std::endl;
        std::cout << "FEED: " << FEED_RATE << std::endl;
      } else if (const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()) {
        sf::Vector2i mousePos = mouseMoved->position;
        int x = mousePos.x;
        int y = mousePos.y;
        arr[y*WIDTH+x].b = 1.0f;
        int stroke = 10;
        for (int i = y-stroke/2; i < y+stroke/2; i++) {
          if (i < 0 || i >= HEIGHT) continue;
          for (int j = x-stroke/2; j < x+stroke/2; j++) {
            if (j < 0 || j >= WIDTH) continue;
            arr[i*WIDTH+j].b = 1.0f;
            arr[i*WIDTH+j].a = 0.0f;
          }
        }
      }
    }
    window.setActive();
    updatearr(arr, nextarr);
    // std::cout << "update" << std::endl;

    // Update the image with the new arr data
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        Cell c = arr[y*WIDTH+x];
        uint8_t value = 255 - static_cast<uint8_t>((c.a-c.b)*255);
        image.setPixel(sf::Vector2u(x, y), sf::Color(value, value, value));
      }
    }
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    window.draw(sprite);
    window.display();
  }

  return 0;
}
