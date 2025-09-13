# JS Frontend Build and Development Tools

## Webpack

```js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  // 1. Entry point of the application
  entry: './src/index.js', 
  
  // 2. Output configuration where the bundled files will be placed
  output: {
    path: path.resolve(__dirname, 'dist'), 
    filename: 'bundle.js', 
    clean: true // Clean up the 'dist' folder before each build
  },
  
  // 3. Module rules to handle different file types
  module: {
    rules: [
      {
        // 4. Rule for JavaScript and JSX files (Transpile ES6+ and JSX to ES5)
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
          },
        },
      },
      {
        // 5. Rule for CSS files
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        // 6. Rule for images (handling images like .png, .jpg, etc.)
        test: /\.(png|svg|jpg|jpeg|gif)$/i,
        type: 'asset/resource',
      },
    ],
  },

  // 7. Plugins (extensions that hook into the Webpack build process)
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html', // Generates index.html and includes the bundle
    }),
  ],

  // 8. Development server configuration
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true, // Enable gzip compression for everything served
    port: 9000, // Serve the application at localhost:9000
    hot: true, // Enable hot module replacement
  },

  // 9. Set the mode for different optimizations (can be 'development', 'production', or 'none')
  mode: 'development',
};

```

* build

`npm run build`.

* `--save` vs `--save-dev`

`--save`: dependencies will be used in production.

`--save-dev`: dependencies are only used during development.

#### General Flow of Webpack

#### Plugins

#### module federation

## Vite

Vite (pronounced "veet," the French word for "quick") is a high-performance equivalent to webpack.