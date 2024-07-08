import React from 'react';
import { ChakraProvider, Box, Heading, Link, VStack, Button, useColorMode, IconButton } from '@chakra-ui/react';
import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import { FaSun, FaMoon } from 'react-icons/fa';
import logo from './assets/o2-awareness-logo.png';
import './App.css';
import EducationalModule from './components/EducationalModule';
import SimulationModule from './components/SimulationModule';
import SustainabilityModule from './components/SustainabilityModule';
import FutureScenarioPlanningTool from './components/FutureScenarioPlanningTool';
import CommunityForum from './components/CommunityForum';
import theme from './theme';

function App() {
  const { colorMode, toggleColorMode } = useColorMode();

  return (
    <ChakraProvider theme={theme}>
      <Router>
        <Box className="App" p={{ base: 4, md: 8 }} maxW="1200px" mx="auto">
          <header className="App-header" mb={{ base: 4, md: 8 }}>
            <img src={logo} className="App-logo" alt="logo" style={{ width: '100%', maxWidth: '200px' }} />
            <Heading as="h1" size={{ base: 'lg', md: 'xl' }} mb={4}>
              Oxygen Agent
            </Heading>
            <IconButton
              aria-label="Toggle dark mode"
              icon={colorMode === 'light' ? <FaMoon /> : <FaSun />}
              onClick={toggleColorMode}
              mb={4}
            />
            <VStack spacing={4} w="full">
              <Button as={RouterLink} to="/educational" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Educational Module
              </Button>
              <Button as={RouterLink} to="/simulation" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Simulation Module
              </Button>
              <Button as={RouterLink} to="/sustainability" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Sustainability Module
              </Button>
              <Button as={RouterLink} to="/future-planning" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Future Scenario Planning Tool
              </Button>
              <Button as={RouterLink} to="/community-forum" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
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
