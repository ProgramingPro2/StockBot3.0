import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Train from "./pages/Train";
import Tune from "./pages/Tune";
import Backtest from "./pages/Backtest";
import Paper from "./pages/Paper";
import Live from "./pages/Live";
import ModelsPage from "./pages/Models";
import SchedulePage from "./pages/Schedule";
import Plots from "./pages/Plots";
import LogsPage from "./pages/Logs";
import TensorBoardPage from "./pages/TensorBoard";
import ConfigPage from "./pages/Config";
import SystemPage from "./pages/System";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="/train" element={<Train />} />
        <Route path="/tune" element={<Tune />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/paper" element={<Paper />} />
        <Route path="/live" element={<Live />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/schedule" element={<SchedulePage />} />
        <Route path="/plots" element={<Plots />} />
        <Route path="/logs" element={<LogsPage />} />
        <Route path="/tensorboard" element={<TensorBoardPage />} />
        <Route path="/config" element={<ConfigPage />} />
        <Route path="/system" element={<SystemPage />} />
      </Route>
    </Routes>
  );
}
