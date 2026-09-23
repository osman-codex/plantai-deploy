import { NavLink, Route, Routes } from "react-router-dom";
import Logo from "./components/Logo";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import Monitor from "./pages/Monitor";
import Research from "./pages/Research";
import About from "./pages/About";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Logo size={36} id="top" />
          <span>PlantGuard <em>AI</em></span>
        </div>
        <nav>
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/analyze">Analyze</NavLink>
          <NavLink to="/monitor">Monitor</NavLink>
          <NavLink to="/research">Research</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/research" element={<Research />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="footer">
        <span className="footer-brand"><Logo size={18} id="ft" /> PlantGuard AI</span>
        {" "}— real trained model, honest predictions.
      </footer>
    </div>
  );
}
