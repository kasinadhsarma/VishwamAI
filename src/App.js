import React from 'react';
import { ChakraProvider, Box, Heading, Link, VStack, Button } from '@chakra-ui/react';
import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import logo from './assets/o2-awareness-logo.png';
import './App.css';
import EducationalModule from './components/EducationalModule';
import SimulationModule from './components/SimulationModule';
import SustainabilityModule from './components/SustainabilityModule';
import FutureScenarioPlanningTool from './components/FutureScenarioPlanningTool';
import CommunityForum from './components/CommunityForum';

function App() {
  return (
    <ChakraProvider>
      <Router>
        <Box className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <Heading as="h1" size="xl" mb={4}>
              Oxygen Agent
            </Heading>
            <VStack spacing={4}>
              <Button as={RouterLink} to="/educational" colorScheme="teal" size="lg">
                Educational Module
              </Button>
              <Button as={RouterLink} to="/simulation" colorScheme="teal" size="lg">
                Simulation Module
              </Button>
              <Button as={RouterLink} to="/sustainability" colorScheme="teal" size="lg">
                Sustainability Module
              </Button>
              <Button as={RouterLink} to="/future-planning" colorScheme="teal" size="lg">
                Future Scenario Planning Tool
              </Button>
              <Button as={RouterLink} to="/community-forum" colorScheme="teal" size="lg">
                Community Forum
              </Button>
            </VStack>
            <p>
              Edit <code>src/App.js</code> and save to reload.
            </p>
            <Link
              className="App-link"
              href="https://reactjs.org"
              isExternal
            >
              Learn React
            </Link>
          </header>
          <Routes>
            <Route path="/educational" element={<EducationalModule />} />
            <Route path="/simulation" element={<SimulationModule />} />
            <Route path="/sustainability" element={<SustainabilityModule />} />
            <Route path="/future-planning" element={<FutureScenarioPlanningTool />} />
            <Route path="/community-forum" element={<CommunityForum />} />
          </Routes>
        </Box>
      </Router>
    </ChakraProvider>
  );
}

export default App;
