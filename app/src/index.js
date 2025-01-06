import React from 'react';
import ReactDOM from 'react-dom';
import App from 'components/App';  // 正しいパスに修正

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
